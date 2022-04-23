import os
import shutil
import requests
import tempfile
import errno
import zipfile
from tqdm.auto import tqdm

# Change this to the path where you want to download the dataset to
DEFAULT_ROOT = 'Developer/bachelorarbeit/data'
URL = 'https://motion-annotation.humanoids.kit.edu/downloads/4/'
# BUFFER_SIZE = 32 * 2048 * 2048


class Motion:

    def __init__(self, meta, format_, annotation, motion_data):
        self.meta = meta
        self.annotation = annotation
        self.format_ = format_
        self.motion_data = motion_data

    def parse(self):
        pass


class MotionDataset:
    urls = [
        'https://motion-annotation.humanoids.kit.edu/downloads/4/',
    ]

    def __init__(self, root=DEFAULT_ROOT, train=True, transform=None):
        self.root = os.path.expanduser(root)
        print(self.root)

    def download(self):
        root = os.path.expanduser(DEFAULT_ROOT)
        print('Downloading the dataset...')
        with requests.get(URL, stream=True) as request:
            print(request.headers)
            dataset_length = int(request.headers.get('Content-Length'))
            with tqdm.wrapattr(
                request.raw,
                'read',
                total=dataset_length,
                desc='',
            ) as raw_data:

                with open(
                    f'{os.path.basename(request.url)}',
                    'wb'
                ) as dataset:
                    shutil.copyfileobj(raw_data, dataset)
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
            'The possible formats are MMM, C3D: '
        )
        os.chdir('data/motion_dataset')
        types, formats = [], []
        for file in os.listdir():
            name, file_format = file.split('.')
            id_, type_ = name.split('_')
            types.append(type_)
            formats.append(file_format)
            if format_ == file_format.lower():
                with open(file, 'r') as file:
                    pass
                print(f'id is {id_}, type is {type_}')
        print(
            f'The types are {set(types)}',
            f'and the formats are {set(formats)}',
        )
        os.chdir(current_directory)


if __name__ == '__main__':
    motion_dataset = MotionDataset()
    motion_dataset.parse()

