import os
import logging
# import json
import xml.etree.cElementTree as ET
import shutil
import requests
import zipfile
from tqdm.auto import tqdm
from robot import Robot

# logging.basicConfig(
#     filename='logs/logger.log',
#     encoding='utf-8',
#     level=logging.DEBUG,
# )
# Change this to the path where you want to download the dataset to
DEFAULT_ROOT = 'data/motion_data'
URL = 'https://motion-annotation.humanoids.kit.edu/downloads/4/'
# BUFFER_SIZE = 32 * 2048 * 2048


class Motion:

    def __init__(self, format_, motion_file, meta=None, annotation=None):
        self.meta = meta
        self.annotation = annotation
        self.format_ = format_
        self.motion_file = motion_file

    '''
    The parsing section of this model is inspired by the code snippet
    in https://motion-annotation.humanoids.kit.edu/dataset/
    '''
    def _parse_list(self, xml_elem, length, indexes=None):

        if indexes is None:
            indexes = range(length)

        elems = [
            float(x) for idx, x in enumerate(xml_elem.text.rstrip().split(' '))
            if idx in indexes
        ]

        if len(elems) != length:
            raise RuntimeError('invalid number of elements')

        return elems

    def _parse_frame(self, xml_frame, joint_indexes):
        xml_joint_pos = xml_frame.find('JointPosition')

        if xml_joint_pos is None:
            raise RuntimeError('<JointPosition> not found')

        joint_pos = self._parse_list(
            xml_joint_pos,
            len(joint_indexes),
            joint_indexes,
        )

        return joint_pos

    def _parse_motion(self, xml_motion):
        xml_joint_order = xml_motion.find('JointOrder')

        if xml_joint_order is None:
            raise RuntimeError('<JointOrder> not found')

        joint_names = []
        joint_indexes = []

        for idx, xml_joint in enumerate(xml_joint_order.findall('Joint')):
            name = xml_joint.get('name')
            if name is None:
                raise RuntimeError('<Joint> has no name')
            joint_indexes.append(idx)
            joint_names.append(name)

        frames = []
        xml_frames = xml_motion.find('MotionFrames')

        if xml_frames is None:
            raise RuntimeError('<MotionFrames> not found')

        for xml_frame in xml_frames.findall('MotionFrame'):
            frames.append(self._parse_frame(xml_frame, joint_indexes))

        xml_config = xml_motion.findall('ModelProcessorConfig')
        xml_model_height = xml_config[0].findall('Height')
        xml_model_mass = xml_config[0].findall('Mass')
        height = float(xml_model_height[0].text)
        mass = float(xml_model_mass[0].text)
        # self.robot = Robot(height, mass)
        self.robot = Robot(1, mass)

        return joint_names, frames

    def parse(self):
        current_directory = os.getcwd()
        print(f'The annotation of the motion is {self.annotation}')
        os.chdir(DEFAULT_ROOT)
        xml_tree = ET.parse(self.motion_file)
        os.chdir(current_directory)
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
            with tqdm.wrapattr(
                request.raw,
                'read',
                total=dataset_size,
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
        with zipfile.ZipFile(dataset, 'r') as zip_file:
            for file in tqdm(zip_file.infolist(), desc=''):
                zip_file.extract(
                    file,
                    os.path.expanduser(DEFAULT_ROOT),
                )
        os.remove(dataset)
        print('Done')

    def parse(self):
        print('Parsing the dataset, and extracting mmm-files...')
        current_directory = os.getcwd()
        format_ = 'mmm'
        os.chdir(DEFAULT_ROOT)
        ids = map(
            lambda x: x.split('_')[0],
            sorted(os.listdir()),
        )
        ids = sorted(list(set(ids)))
        for id_ in ids:
            with open(f'{id_}_annotations.json', 'r',) as file:
                annotation = file.read()
            with open(f'{id_}_meta.json', 'r') as file:
                meta = file.read()
            self.motions.append(
                Motion(
                    format_,
                    f'{id_}_{"mmm.xml" if format_ == "mmm" else "raw.c3d"}',
                    annotation=annotation,
                    meta=meta,
                )
            )
        os.chdir(current_directory)
