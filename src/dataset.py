import os
from enum import Enum
from contextlib import redirect_stdout
import numpy as np
from tqdm import tqdm
from tqdm.auto import tqdm as download_wrapper
import logging
import json
import xml.etree.cElementTree as ET
import shutil
import requests
import zipfile
from nltk import download, FreqDist, pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import EnglishStemmer
from string import punctuation
from torch import load, save
from torch.utils.data import random_split

from utils import change_to, prepare_dataset, activities_dictionary
from plotters import (
    visualize_length_distribution,
    visualize_class_distribution,
)

# Change this to the path where you want to download the dataset to
DEFAULT_ROOT = 'data/motion_data'
# BUFFER_SIZE = 32 * 2048 * 2048
URL = 'https://motion-annotation.humanoids.kit.edu/downloads/4/'


class Classification(Enum):
    BASIC = 1
    LABELED = 2
    MULTI_LABELED = 3


class Frame:

    def __init__(
        self,
        timestep,
        root_position,
        root_rotation,
        joints,
        joint_positions,
        joint_velocities,
        joint_accelerations,
    ):
        self.timestep = timestep
        self.joints = joints
        self.root_position = root_position
        self.root_rotation = root_rotation
        self.joint_positions = joint_positions
        self.joint_velocities = joint_velocities
        self.joint_accelerations = joint_accelerations
        self.parse()

    def parse(self):
        self.parse_root_position()
        self.parse_root_rotation()
        self.parse_joint_positions()
        self.parse_joint_velocities()
        self.parse_joint_accelerations()

    @staticmethod
    def _parse_list(lst):
        return list(map(lambda x: float(x), lst.rstrip().split(' ')))

    def parse_root_position(self):
        self.root_position = self._parse_list(self.root_position)

    def parse_root_rotation(self):
        self.root_rotation = self._parse_list(self.root_rotation)

    def parse_joint_positions(self):
        self.joint_positions = {
            joint: position
            for joint, position
            in zip(self.joints, self._parse_list(self.joint_positions))
        }

    def parse_joint_velocities(self):
        self.joint_velocities = {
            joint: velocity
            for joint, velocity
            in zip(self.joints, self._parse_list(self.joint_velocities))
        }

    def parse_joint_accelerations(self):
        self.joint_accelerations = {
            joint: acceleration
            for joint, acceleration
            in zip(self.joints, self._parse_list(self.joint_accelerations))
        }


class Motion:

    def __init__(self, format_, motion_file, id_, annotation=None):
        self.id_ = id_
        self.annotation = annotation
        self.format_ = format_
        self.motion_file = motion_file
        self.classification = None

    def get_label(self, labels, stop_stems):
        try:
            words = word_tagger(word_tokenize(self.annotation))
            for word_tag in words:
                if word_tag[1] in ['VB', 'VBZ', 'VBG']:
                    stem_ = stem(word_tag[0])
                    if (
                        stem_ not in stop_stems and
                        stem_ in labels
                    ):
                        self.classification = stem_
        except KeyError:
            # TODO create new Exception for unlabelable motions
            raise KeyError

    def get_extended_label(
        self,
        labels,
        stop_stems,
        classes,
        collect_all=False,
    ):
        try:
            words = word_tagger(word_tokenize(self.annotation))
            verbs = set()
            for word_tag in words:
                if word_tag[1] in ['VB', 'VBZ', 'VBG']:
                    stem_ = stem(word_tag[0])
                    if collect_all:
                        if stem_ not in stop_stems:
                            verbs.add(stem(word_tag[0]))
                    else:
                        if stem_ not in stop_stems and stem_ in labels:
                            verbs.add(stem(word_tag[0]))
            self.classification = ' '.join(verbs)
            classes.add(' '.join(verbs))
        except KeyError:
            # TODO create new Exception for unlabelable motions
            raise KeyError

    def classify_motion(
        self,
        labels,
        basic=False,
        extended_labeling=False,
        classes=None,
    ):
        stop_stems = [
            'is',
            'go',
            'keep',
            'start',
            'do',
            'continu',
            'get',
            'perform',
            'has',
            'look',
            'goe',
        ]
        if basic:
            self.get_label(labels, stop_stems)

        elif extended_labeling:
            self.get_extended_label(labels, stop_stems, classes)

    def _parse_frame(self, xml_frame, joints):
        return Frame(
            float(xml_frame.find('Timestep').text),
            xml_frame.find('RootPosition').text,
            xml_frame.find('RootRotation').text,
            joints,
            xml_frame.find('JointPosition').text,
            xml_frame.find('JointVelocity').text,
            xml_frame.find('JointAcceleration').text,
        )

    def _parse_motion(self, xml_motion):
        xml_joint_order = xml_motion.find('JointOrder')

        if xml_joint_order is None:
            raise RuntimeError('<JointOrder> not found')

        joints = []
        for idx, xml_joint in enumerate(xml_joint_order.findall('Joint')):
            name = xml_joint.get('name')
            if name is None:
                raise RuntimeError('<Joint> has no name')
            joints.append(name)

        xml_frames = xml_motion.find('MotionFrames')
        if xml_frames is None:
            raise RuntimeError('<MotionFrames> not found')
        self.frames = []
        for index, xml_frame in enumerate(xml_frames.findall('MotionFrame')):
            self.frames.append(self._parse_frame(xml_frame, joints))

        xml_config = xml_motion.findall('ModelProcessorConfig')
        xml_model_height = xml_config[0].findall('Height')
        xml_model_mass = xml_config[0].findall('Mass')
        self.scale_factor = float(xml_model_height[0].text)
        self.mass = float(xml_model_mass[0].text)

        del xml_config
        del xml_joint_order
        del xml_model_height
        del xml_model_mass
        del xml_frames

        return joints, self.frames

    @change_to(DEFAULT_ROOT)
    def parse(self):
        with open(self.motion_file, 'r') as motion_file:
            xml_tree = ET.parse(motion_file)
            xml_root = xml_tree.getroot()
            self.xml_motions = xml_root.findall('Motion')
            self.motions = []

            if len(self.xml_motions) > 1:
                logging.warn(
                    'more than one <Motion> tag in file "%s", '
                    'only parsing the first one'
                )

            for motion in self.xml_motions:
                self.motions.append(self._parse_motion(motion))
            del xml_tree
            del xml_root
            del self.xml_motions

    @change_to(DEFAULT_ROOT)
    def get_matrixified_joint_positions(
        self,
        frequency=1,
        max_length=None,
        min_length=None,
        inverse=True,
        padding='zero',
    ):
        try:
            self.position_matrix = np.load(f'{self.id_}_joint_positions.npy')
        except FileNotFoundError:
            self.matrixfy_all()
        if max_length and self.position_matrix.shape[0] > max_length:
            return (
                self.position_matrix[:max_length:frequency],
                self.classification
            )
        if min_length and self.position_matrix.shape[0] < min_length:
            if padding == 'zero':
                padding = np.zeros(
                    (
                        min_length - self.position_matrix.shape[0],
                        self.position_matrix.shape[1],
                    )
                )
            elif padding == 'last':
                padding = np.tile(
                    self.position_matrix[-1, :],
                    (min_length - self.position_matrix.shape[0], 1),
                )
            else:
                raise ValueError
            self.position_matrix = np.vstack((self.position_matrix, padding))
        if inverse:
            return (
                np.flip(self.position_matrix[::frequency]),
                self.classification,
            )
        else:
            return self.position_matrix[::frequency], self.classification

    @change_to(DEFAULT_ROOT)
    def get_matrixified_root_infomation(
        self,
        frequency=1,
        max_length=None,
        min_length=None,
        inverse=True,
        padding='zero',
    ):
        try:
            root_positions = np.load(f'{self.id_}_root_positions.npy')
            root_rotations = np.load(f'{self.id_}_root_rotations.npy')
        except FileNotFoundError:
            self.matrixfy_all()
        self.position_matrix = np.hstack(
            (root_positions, root_rotations)
        )
        if max_length and self.position_matrix.shape[0] > max_length:
            return (
                self.position_matrix[:max_length:frequency],
                self.classification
            )
        if min_length and self.position_matrix.shape[0] < min_length:
            if padding == 'zero':
                padding = np.zeros(
                    (
                        min_length - self.position_matrix.shape[0],
                        self.position_matrix.shape[1],
                    )
                )
            elif padding == 'last':
                padding = np.tile(
                    self.position_matrix[-1, :],
                    (min_length - self.position_matrix.shape[0], 1),
                )
            else:
                raise ValueError
            self.position_matrix = np.vstack(
                (self.position_matrix, padding)
            )
        if inverse:
            return (
                np.flip(self.position_matrix[::frequency]),
                self.classification,
            )

        else:
            return self.position_matrix[::frequency], self.classification

    @change_to(DEFAULT_ROOT)
    def get_matrixified_all(
        self,
        frequency=1,
        max_length=None,
        min_length=None,
        inverse=True,
        padding='zero',
    ):
        try:
            root_positions = np.load(f'{self.id_}_root_positions.npy')
            root_rotations = np.load(f'{self.id_}_root_rotations.npy')
            joint_positions = np.load(f'{self.id_}_joint_positions.npy')
        except FileNotFoundError:
            self.matrixfy_all()
        self.position_matrix = np.hstack(
            (root_positions, root_rotations, joint_positions)
        )
        if max_length and self.position_matrix.shape[0] > max_length:
            return (
                self.position_matrix[:max_length:frequency],
                self.classification
            )
        if min_length and self.position_matrix.shape[0] < min_length:
            if padding == 'zero':
                padding = np.zeros(
                    (
                        min_length - self.position_matrix.shape[0],
                        self.position_matrix.shape[1],
                    )
                )
            elif padding == 'last':
                padding = np.tile(
                    self.position_matrix[-1, :],
                    (min_length - self.position_matrix.shape[0], 1),
                )
            else:
                raise ValueError
            self.position_matrix = np.vstack(
                (self.position_matrix, padding)
            )
        if inverse:
            return (
                np.flip(self.position_matrix[::frequency]),
                self.classification,
            )
        else:
            return self.position_matrix[::frequency], self.classification

    def matrixfy_all(self):
        self.parse()
        self.matrixfy_frames()
        self.matrixfy_root_positions()
        self.matrixfy_root_rotations()
        del self.frames

    def matrixfy_frames(self):
        np.save(
            f'{self.id_}_joint_positions.npy',
            np.array(
                [list(frame.joint_positions.values()) for frame in self.frames]
            ),
        )

    def matrixfy_root_positions(self):
        np.save(
            f'{self.id_}_root_positions.npy',
            np.array(
                [frame.root_position for frame in self.frames]
            ),
        )

    def matrixfy_root_rotations(self):
        np.save(
            f'{self.id_}_root_rotations.npy',
            np.array(
                [frame.root_rotation for frame in self.frames]
            ),
        )

    def get_initial_root_position(self):
        return self.frames[0].root_position


class MotionDataset:
    urls = ['https://motion-annotation.humanoids.kit.edu/downloads/4/']

    def __init__(
        self,
        root=DEFAULT_ROOT,
        train=False,
        classification=Classification(1),
        labels=None,
        get_matrixified_root_infomation=False,
        get_matrixified_joint_positions=False,
        get_matrixified_all=False,
        frequency=1,
        max_length=None,
        min_length=None,
        padding='zero',
        inverse=True,
    ):
        self.root = os.path.expanduser(root)
        self.train = train
        self.labels = labels
        self.motions = []
        self.classification = classification
        self.frequency = frequency
        self.max_length = max_length
        self.min_length = min_length
        self.get_matrixified_root_infomation = get_matrixified_root_infomation
        self.get_matrixified_joint_positions = get_matrixified_joint_positions
        self.get_matrixified_all = get_matrixified_all
        self.inverse = inverse
        self.padding = padding

    def download(self):
        print('Downloading the dataset...')
        with requests.get(URL, stream=True) as request:
            dataset_size = int(request.headers.get('Content-Length'))
            with download_wrapper.wrapattr(
                request.raw,
                'read',
                total=dataset_size,
                ncols=100,
                desc='',
            ) as raw_data:
                with open(f'{os.path.basename(request.url)}', 'wb') as dataset:
                    shutil.copyfileobj(raw_data, dataset)
        return dataset

    def extract(self):
        dataset = self.download()
        os.makedirs(DEFAULT_ROOT)
        print('Extracting the dataset...')
        with zipfile.ZipFile(dataset.name, 'r') as zip_file:
            for file in tqdm(zip_file.infolist(), desc='', ncols=100,):
                zip_file.extract(
                    file,
                    os.path.expanduser(DEFAULT_ROOT),
                )
        dataset.close()
        os.remove(dataset.name)

    def parse_motion(self, id_, format_='mmm'):
        if 'D' in id_ or id_ == 'mapping.csv':
            return None
        json_file = open(f'{id_}_annotations.json', 'r')
        annotation = ' '.join(json.load(json_file))
        if annotation == '':
            json_file.close()
            os.remove(f'{id_}_meta.json')
            os.remove(f'{id_}_annotations.json')
            os.remove(f'{id_}_mmm.xml')
            os.remove(f'{id_}_raw.c3d')
            return None
        json_file.close()
        motion = Motion(
            format_,
            f'{id_}_{"mmm.xml" if format_ == "mmm" else "raw.c3d"}',
            id_,
            annotation=annotation,
        )
        if not os.path.exists(f'{id_}_joint_positions.txt',) or\
           not os.path.exists(f'{id_}_joint_rotations.txt',) or\
           not os.path.exists(f'{id_}_root_positions.txt'):
            motion.matrixfy_all()

        try:
            if self.classification == Classification.BASIC:
                motion.classify_motion(self.labels, basic=True)
            elif self.classification == Classification.LABELED:
                motion.classify_motion(
                    self.labels,
                    extended_labeling=True,
                    classes=self.classes,
                )
        except KeyError:
            self.miss_classification += 1

        if self.get_matrixified_joint_positions:
            matrix_representation, label = motion.get_matrixified_joint_positions(
                frequency=self.frequency,
                max_length=self.max_length,
                min_length=self.min_length,
                inverse=self.inverse,
                padding=self.padding,
            )
            if label:
                self.matrix_represetations.append(
                    (matrix_representation, label)
                )
        if self.get_matrixified_root_infomation:
            matrix_representation, label = motion.get_matrixified_root_infomation(
                frequency=self.frequency,
                max_length=self.max_length,
                min_length=self.min_length,
                inverse=self.inverse,
                padding=self.padding,
            )
            if label:
                self.matrix_represetations.append(
                    (matrix_representation, label)
                )
        if self.get_matrixified_all:
            matrix_representation, label = motion.get_matrixified_all(
                frequency=self.frequency,
                max_length=self.max_length,
                min_length=self.min_length,
                inverse=self.inverse,
                padding=self.padding,
            )
            if label:
                self.matrix_represetations.append(
                    (matrix_representation, label)
                )

        return motion

    @change_to(DEFAULT_ROOT)
    def parse(self):
        format_ = 'mmm'
        if os.listdir('.') == []:
            self.extract()
        print(
            'Parsing the dataset, removing motions without annotations '
            'and extracting mmm-files...'
        )
        ids = sorted(
            list(
                set(
                    map(
                        lambda x: x.split('_')[0],
                        sorted(os.listdir()),
                    )
                )
            )
        )
        self.miss_classification = 0
        self.classes = set()
        self.matrix_represetations = []

        for id_ in tqdm(ids, ncols=100,):
            motion = self.parse_motion(id_, format_)
            if motion:
                self.motions.append(motion)


def tokenize_annotation(annotation):
    with redirect_stdout(open(os.devnull, "w")):
        if not os.path.exists(
            os.path.expanduser('~/nltk_data/tokenizers/punkt')
        ):
            download('punkt')
        if not os.path.exists(
            os.path.expanduser('~/nltk_data/corpora/stopwords')
        ):
            download('stopwords')
        stop_words = set(stopwords.words('english'))
        tokens = word_tokenize(annotation)
        return [token for token in tokens if token not in stop_words]


def classify_annotation(
    annotation,
    activity,
    similarity_threshhold=.6,
    get_hits=False,
):
    with redirect_stdout(open(os.devnull, "w")):
        tokens = tokenize_annotation(annotation)
        similarities = []

        if not os.path.exists(os.path.expanduser('~/nltk_data/corpora')):
            download('wordnet')
        if not os.path.exists(os.path.expanduser('~/nltk_data/corpora')):
            download('omw-1.4')

        for token in tokens:
            synonyms = wordnet.synsets(token)
            synonym_similarity = [
                activity.wup_similarity(synonym)
                for synonym in synonyms
                if activity.wup_similarity(synonym) > similarity_threshhold
            ]
            try:
                synonym_similarity = max(synonym_similarity)
            except ValueError:
                synonym_similarity = 0
            similarities.append(synonym_similarity)

        if sum(similarities) > .5:
            return annotation


def remove_ponctuations(annotation_text):
    plain_text = ''
    for word in tokenize_annotation(annotation_text):
        if word not in punctuation:
            plain_text += ' ' + word
    for punctuation_ in punctuation:
        plain_text.replace(punctuation, ' ')
    return plain_text


def lemmetize_annotation(tokenized_annotation_text):
    lemmatizer = WordNetLemmatizer()
    return [
        lemmatizer.lemmatize(
            remove_ponctuations(
                word
            )
        ) for word in tokenized_annotation_text
    ]


def stem(word):
    Stemmmer = EnglishStemmer()
    return Stemmmer.stem(word)


def word_tagger(tokenized_annotation_text):
    if not os.path.exists(
        os.path.expanduser('~/nltk_data/taggers/averaged_perceptron_tagger')
    ):
        download('averaged_perceptron_tagger')
    return pos_tag(tokenized_annotation_text)


def get_most_common_verbs(annotation_text, number_verbs=0):
    set_ = set()
    for item in FreqDist(
        word_tagger(word_tokenize(annotation_text))
    ).most_common():
        if item[0][1] in ['VB', 'VBD', 'VBZ', 'VBG']:
            set_.add(item)
    stemmed_verbs = [(stem(verb), frequency) for (verb, _), frequency in set_]
    dict_stemmed_verbs = {}
    for stem_, frequency in stemmed_verbs:
        if stem_ in dict_stemmed_verbs.keys():
            dict_stemmed_verbs[stem_] += frequency
        else:
            dict_stemmed_verbs[stem_] = frequency
    if number_verbs == 0:
        return dict_stemmed_verbs.keys()
    else:
        return sorted(
            dict_stemmed_verbs.items(), key=lambda x: x[1], reverse=True,
        )[:number_verbs]


def get_dataset_infos(dataset):
    print(
        f'Classification miss-rate is '
        f'{dataset.miss_classification / len(dataset.motions)}'
        f'The number of classes is {len(dataset.classes)}',
    )
    print(
        f'The number of miss classification is {dataset.miss_classification}'
    )
    annotations = [''.join(motion.annotation) for motion in dataset.motions]
    percentage_annotated_motions = (
        len(list(annotation for annotation in annotations if annotation)) /
        len(annotations)
    ) * 100
    print(
        f'{percentage_annotated_motions:.2f} is the percentatge of motions '
        f'with annotations'
    )
    print(get_most_common_verbs(' '.join(annotations), number_verbs=30))


def get_number_infos_motions(dataset):
    lengths = {}
    number_compatible_motions = 0
    number_motions_under_20 = 0
    number_motions_under_10 = 0
    number_motions_under_5 = 0

    dataset.parse()
    print('Get the lengths of the frames...')
    for motion in tqdm(dataset.motions, ncols=100,):
        motion.parse()
        duration = len(motion.frames)
        lengths[motion.id_] = duration
        del motion.frames
        if duration / 1000 > 20:
            number_compatible_motions += 1
        elif 10 < duration / 1000 <= 20:
            number_motions_under_20 += 1
        elif 5 < duration / 1000 <= 10:
            number_motions_under_10 += 1
        elif duration / 1000 <= 5:
            number_motions_under_5 += 1

    visualize_length_distribution(lengths)
    print(
        f'Max length is {max(lengths)}',
        f'Min length is {min(lengths)}',
        f'The number of lengths of frames is {len(lengths)}',
        f'The number of motions > 20 is {number_compatible_motions}',
        f'The number of motions <= 20 is {number_motions_under_20}',
        f'The number of motions <= 10 is {number_motions_under_10}',
        f'The number of motions > 5 is {number_motions_under_5}',
        sep='\n',
    )


def create_dataset(
    labels=None,
    train=True,
    frequency=None,
    min_length=None,
    max_length=None,
    get_matrixified_root_infomation=False,
    get_matrixified_joint_positions=False,
    get_matrixified_all=False,
    padding=None,
    inverse=False,
    normalize=True,
    oversample=True,
):
    dataset = MotionDataset(
        train=train,
        classification=Classification(2),
        labels=labels,
        get_matrixified_root_infomation=get_matrixified_root_infomation,
        get_matrixified_joint_positions=get_matrixified_joint_positions,
        get_matrixified_all=get_matrixified_all,
        frequency=frequency,
        max_length=max_length,
        min_length=min_length,
        padding=padding,
        inverse=inverse,
    )
    try:
        dataset.parse()
    except FileNotFoundError:
        dataset.extract()
        dataset.parse()
    dataset = dataset.matrix_represetations
    label_frequency = visualize_class_distribution(dataset)
    power = 1.2
    num_datapoints = sum(
        frequency for label, frequency in label_frequency.items()
        if label in labels
    )
    oversample_values = {
        label: (
            1
            if not oversample or label == 'walk'
            else label_frequency['walk'] // label_frequency[label] // 2
        )
        for label, frequency in label_frequency.items()
        if label in labels
    }
    label_ratio = {
        label: frequency ** power / num_datapoints
        for label, frequency in label_frequency.items()
        if label in labels
    }
    label_ratio = {
        label: label_ratio[label] for label in labels
    }
    with open('label_ratio.json', 'w') as json_file:
        json.dump(label_ratio, json_file)
    dataset = prepare_dataset(
        dataset,
        labels,
        normalize=normalize,
        oversample=oversample,
        oversample_values=oversample_values,
    )
    save(dataset, 'dataset.pt')
    return dataset


def load_dataset(
    labels=None,
    train=True,
    frequency=None,
    min_length=None,
    max_length=None,
    get_matrixified_root_infomation=False,
    get_matrixified_joint_positions=False,
    get_matrixified_all=False,
    padding=None,
    inverse=False,
    normalize=True,
    oversample=True,
):

    print('Loading the dataset...')
    try:
        dataset = load('dataset.pt')
    except FileNotFoundError:
        dataset = create_dataset(
            labels=labels,
            train=train,
            get_matrixified_root_infomation=get_matrixified_root_infomation,
            get_matrixified_joint_positions=get_matrixified_joint_positions,
            get_matrixified_all=get_matrixified_all,
            frequency=frequency,
            max_length=max_length,
            min_length=min_length,
            padding=padding,
            inverse=inverse,
            normalize=normalize,
            oversample=oversample,
        )
    with open('label_ratio.json', 'r') as json_file:
        label_ratio = json.load(json_file)
    weights = [ratio for label, ratio in label_ratio.items()]
    return weights, dataset


def get_folds(
    num_folds=None,
    labels=None,
    frequency=None,
    min_length=None,
    max_length=None,
    get_matrixified_root_infomation=False,
    get_matrixified_joint_positions=False,
    get_matrixified_all=False,
    padding=None,
    inverse=False,
    normalize=True,
    oversample=True,
):

    weights, dataset = load_dataset(
        labels=labels,
        train=bool(num_folds),
        frequency=frequency,
        min_length=min_length,
        max_length=max_length,
        get_matrixified_root_infomation=get_matrixified_root_infomation,
        get_matrixified_joint_positions=get_matrixified_joint_positions,
        get_matrixified_all=get_matrixified_all,
        padding=padding,
        inverse=inverse,
        normalize=normalize,
        oversample=oversample,
    )

    folds = random_split(dataset, [1/num_folds]*num_folds)

    return weights, folds


if __name__ == '__main__':
    dataset = MotionDataset(
        labels=activities_dictionary,
    )
    get_number_infos_motions(dataset)
