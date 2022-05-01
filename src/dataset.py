import os
import logging
import json
import xml.etree.cElementTree as ET
import shutil
import requests
import zipfile
from tqdm.auto import tqdm

# Change this to the path where you want to download the dataset to
DEFAULT_ROOT = 'Developer/bachelorarbeit/data'
URL = 'https://motion-annotation.humanoids.kit.edu/downloads/4/'
# BUFFER_SIZE = 32 * 2048 * 2048


class Motion:

    def __init__(self, format_, motion_data, meta=None, annotation=None):
        self.meta = meta
        self.annotation = annotation
        self.format_ = format_
        self.motion_data = motion_data

    '''
    The parsiong section of this model is inspired by the code snippet
    in https://motion-annotation.humanoids.kit.edu/dataset/
    '''
    def _parse_frame(self, joint_indexes):
        n_joints = len(joint_indexes)
        xml_joint_pos = self.motion_data.find('JointPosition')
        if xml_joint_pos is None:
            raise RuntimeError('<JointPosition> not found')
        joint_pos = self._parse_list(xml_joint_pos, n_joints, joint_indexes)

        return joint_pos

    def _parse_motion(self, motion):
        xml_joint_order = self.motion_data.find('JointOrder')
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
        xml_frames = self.motion_data.find('MotionFrames')

        if xml_frames is None:
            raise RuntimeError('<MotionFrames> not found')

        for xml_frame in xml_frames.findall('MotionFrame'):
            frames.append(self._parse_frame(xml_frame, joint_indexes))

        return joint_names, frames

    def parse(self):
        xml_tree = ET.parse(self.motion_data)
        xml_root = xml_tree.getroot()
        xml_motions = xml_root.findall('Motion')
        motions = []

        if len(xml_motions) > 1:
            logging.warn('more than one <Motion> tag in file "%s", only parsing the first one')

        motions.append(self._parse_motion(xml_motions[0]))
        return motions


class MotionDataset:
    urls = [
        'https://motion-annotation.humanoids.kit.edu/downloads/4/',
    ]

    def __init__(self, root=DEFAULT_ROOT, train=True):
        self.root = os.path.expanduser(root)
        self.train = train
        print(self.root)

    def download(self):
        root = os.path.expanduser(DEFAULT_ROOT)
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
        return dataset, root

    def extract(self):
        dataset, root = self.download()
        print('Extracting the dataset...')
        with zipfile.ZipFile(dataset, 'r') as zip_file:
            zip_file.extract(
                os.path.join(
                    os.expanduser(DEFAULT_ROOT),
                    '2017..',
                ),
                path=root,
            )
        print('Done')

    def parse(self):
        print('Parsing the dataset...')
        current_directory = os.getcwd()
        format_ = input(
            'Enter the type of format type you want to use. '
            'The possible formats are MMM, RAW: '
        )
        os.chdir('data/motion_dataset')
        motions = []
        ids = map(
            lambda x: x.split('_')[0],
            sorted(os.listdir()),
        )
        for id_ in ids:
            with open(f'{id_}_annotations.json', 'r',) as file:
                annotation = file.read()
            with open(f'{id_}_meta.json', 'r') as file:
                meta = file.read()
            with open(
                f'{id_}_{format_.lower()}.{"xml" if format_ == "mmm" else "c3d"}',
                'rb',
            ) as file:
                motion_data = file.read()
            motions.append(
                Motion(
                    format_,
                    motion_data,
                    annotation=annotation,
                    meta=meta,
                )
            )
        os.chdir(current_directory)


if __name__ == '__main__':
    motion_dataset = MotionDataset()
    motion_dataset.parse()
