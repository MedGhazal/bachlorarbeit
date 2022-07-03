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

        # xml_config = xml_motion.findall('ModelProcessorConfig')
        # xml_model_height = xml_config[0].findall('Height')
        # xml_model_mass = xml_config[0].findall('Mass')
        # height = float(xml_model_height[0].text)
        # mass = float(xml_model_mass[0].text)
        # self.robot = Robot(height, mass)
        # self.robot = Robot(1, mass)

        return joints, self.frames

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
