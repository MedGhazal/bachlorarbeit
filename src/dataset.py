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

    def __init__(self, format_, motion_data, meta=None, annotation=None):
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
