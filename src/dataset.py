import torch
import torchvision.transforms as transforms
import os
import errno
import zipfile

# Change this to the path where you want to download the dataset to
DEFAULT_ROOT = '/data/dataset'


class MotionDataset(torch.utils.data.Dataset):
    urls = [
        'https://motion-annotation.humanoids.kit.edu/downloads/4/',
    ]

    def __init__(self, root=DEFAULT_ROOT, train=True, transform=None):
        self.root = os.path.expanduser(root)
        self.train = train  # training set or test set
        self.transform = transform
        self.download()

        if self.train:
            self.train_data = torch.load(os.path.join(self.root, 'training_data.pt'))
            self.train_labels = torch.load(os.path.join(self.root, 'training_labels.pt'))
        else:
            self.test_data = torch.load(os.path.join(self.root, 'test_data.pt'))
            try:
                self.test_labels = torch.load(os.path.join(self.root, 'test_labels.pt'))
            except FileNotFoundError as e:
                print(f'No test labels found at {e.filename}!')

                class Dummy:
                    def __getitem__(self, item):
                        return None
                self.test_labels = Dummy()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, 'training_data.pt')) and \
            os.path.exists(os.path.join(self.root, 'test_data.pt')) and \
            os.path.exists(os.path.join(self.root, 'training_labels.pt'))

    def download(self):
        from six.moves import urllib

        if self._check_exists():
            return

        # download files
        try:
            os.makedirs(self.root)
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        for url in self.urls:
            print('Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = 'strange_symbols.tar.gz'
            file_path = os.path.join(self.root, filename)
            with open(file_path, 'wb') as f:
                f.write(data.read())
            zipfile.open(file_path, mode='r:gz').extractall(self.root)
            os.unlink(file_path)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        return fmt_str


def to_float(x):
    x = x.float()
    x /= 255
    x *= 2
    x -= 1
    return x.unsqueeze(-3)


DEFAULT_TRANSFORMS = transforms.Lambda(to_float)


def get_strange_symbols_train_loader(batch_size, transform=DEFAULT_TRANSFORMS):
    trainset = MotionDataset(root=DEFAULT_ROOT, train=True,  transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    return trainloader


def get_strange_symbols_train_data(root=DEFAULT_ROOT, transform=DEFAULT_TRANSFORMS):
    return MotionDataset(root=root, train=True, transform=transform)[:]


def get_strange_symbols_test_data(root=DEFAULT_ROOT, transform=DEFAULT_TRANSFORMS):
    return MotionDataset(root=root, train=False, transform=transform)[:]
