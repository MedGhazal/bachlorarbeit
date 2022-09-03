import os
from tqdm import tqdm
from tqdm.auto import tqdm as download_wrapper
import logging
import json
import xml.etree.cElementTree as ET
import shutil
import requests
import zipfile
from nltk import download
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from string import punctuation

from utils import change_to

# Change this to the path where you want to download the dataset to
DEFAULT_ROOT = 'data/motion_data'
URL = 'https://motion-annotation.humanoids.kit.edu/downloads/4/'
activities_dictionary = {
    'walk': 'walk.v.01',
    'dance': 'dance.v.01',
    'swim': 'swim.v.01',
    'stand': 'stand.v.01',
    'run': 'run.v.01',
    'jump': 'jump.v.01',
}
# BUFFER_SIZE = 32 * 2048 * 2048


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
        return list(map(
            lambda x: float(x),
            lst.rstrip().split(' '),
        ))

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

    def __init__(self, format_, motion_file, meta=None, annotation=None):
        self.meta = meta
        self.annotation = annotation
        self.format_ = format_
        self.motion_file = motion_file

    def classify_motion(self, core_activity):
        if 'walk' in ''.join(self.annotation):
            core_activity['walk'].append(self)
        elif 'jump' in ''.join(self.annotation):
            core_activity['jump'].append(self)
        elif 'run' in ''.join(self.annotation):
            core_activity['run'].append(self)
        elif 'dance' in ''.join(self.annotation):
            core_activity['dance'].append(self)
        elif 'sit' in ''.join(self.annotation):
            core_activity['sit'].append(self)
        elif 'swim' in ''.join(self.annotation):
            core_activity['swim'].append(self)
        elif 'standing up' in ''.join(self.annotation):
            core_activity['standing up'].append(self)
        else:
            core_activity['none'].append(self)

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

        self.frames = []
        xml_frames = xml_motion.find('MotionFrames')

        if xml_frames is None:
            raise RuntimeError('<MotionFrames> not found')

        for xml_frame in xml_frames.findall('MotionFrame'):
            self.frames.append(self._parse_frame(xml_frame, joints))

        xml_config = xml_motion.findall('ModelProcessorConfig')
        xml_model_height = xml_config[0].findall('Height')
        xml_model_mass = xml_config[0].findall('Mass')
        self.scale_factor = float(xml_model_height[0].text)
        self.mass = float(xml_model_mass[0].text)

        return joints, self.frames

    @change_to(DEFAULT_ROOT)
    def parse(self):
        xml_tree = ET.parse(self.motion_file)
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

    def get_initial_root_position(self):
        return self.frames[0].root_position


class MotionDataset:
    urls = [
        'https://motion-annotation.humanoids.kit.edu/downloads/4/',
    ]

    def __init__(self, root=DEFAULT_ROOT, train=False):
        self.root = os.path.expanduser(root)
        self.train = train
        self.motions = []

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
                with open(
                    f'{os.path.basename(request.url)}',
                    'wb'
                ) as dataset:
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

    @change_to(DEFAULT_ROOT)
    def parse(self):
        print('Parsing the dataset, and extracting mmm-files...')
        format_ = 'mmm'
        ids = map(
            lambda x: x.split('_')[0],
            sorted(os.listdir()),
        )
        ids = sorted(list(set(ids)))
        self.core_activity = {
            'walk': [],
            'jump': [],
            'run': [],
            'dance': [],
            'sit': [],
            'swim': [],
            'standing up': [],
            'none': [],
        }

        for id_ in tqdm(ids, ncols=100,):
            if 'D' in id_:
                continue
            with open(f'{id_}_annotations.json', 'r') as file:
                annotation = ''.join(json.load(file))
            with open(f'{id_}_meta.json', 'r') as file:
                meta = file.read()
            motion = Motion(
                format_,
                f'{id_}_{"mmm.xml" if format_ == "mmm" else "raw.c3d"}',
                annotation=annotation,
                meta=meta,
            )
            motion.classify_motion(self.core_activity)
            self.motions.append(motion)

    def classify_motions(self):
        self.annotations_classification = {
            key: list(
                filter(
                    lambda x: x,
                    list(
                        classify_annotation(
                            motion.annotation,
                            wordnet.synset(
                                value,
                            ),
                        ) for motion in tqdm(
                            self.motions,
                            ncols=100,
                            desc=f'Extracting for {key}',
                        )
                    ),
                ),
            ) for key, value in activities_dictionary.items()
        }


def tokenize_annotation(annotation):
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
    return [
        token for token in tokens if token not in stop_words
    ]


def classify_annotation(
    annotation,
    activity,
    similarity_threshhold=.6,
    get_hits=False,
):
    tokens = tokenize_annotation(annotation)
    similarities = []
    if not os.path.exists(
        os.path.expanduser('~/nltk_data/corpora')
    ):
        download('wordnet')
    if not os.path.exists(
        os.path.expanduser('~/nltk_data/corpora')
    ):
        download('omw-1.4')

    for token in tokens:
        synonyms = wordnet.synsets(token)
        synonym_similarity = [
            activity.wup_similarity(
                synonym
            ) for synonym in synonyms if activity.wup_similarity(
                synonym
            ) > similarity_threshhold
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
    return plain_text


if __name__ == '__main__':
    dataset = MotionDataset()

    try:
        dataset.parse()
    except FileNotFoundError:
        dataset.extract()
        dataset.parse()

    for activity, motions in dataset.core_activity.items():
        print(activity, len(motions))
    annotations = [
        ' '.join(
            motion.annotation
        ) for motion in dataset.motions
    ]
    # dataset.classify_motions()
    # print(dataset.annotations_classification)
    percentage_annotated_motions = (
        len(list(annotation for annotation in annotations if annotation)) /
        len(annotations)
    ) * 100
    print(
        f'{percentage_annotated_motions:.2f} is the percentatge of motions '
        f'with annotations'
    )
    motion_texts = ''
    for motion in dataset.motions:
        motion_texts += motion.annotation
    tokanized_motion_texts = tokenize_annotation(remove_ponctuations(motion_texts))
    lemmatizer = WordNetLemmatizer()
    lemmatized_motion_texts = lemmatizer.lemmatize(
        ' '.join(tokanized_motion_texts)
    )
    print(lemmatized_motion_texts)
