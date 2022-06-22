import os
import shutil
from dataset import DEFAULT_ROOT
from functools import partial


def check_exists():
    return os.path.isdir(
        os.path.join(
            os.path.expanduser(''),
            'motion_dataset')
    )


def pseudo_wrapper(function, arguments=arguments):
    def func(*args, **kwargs):
        root = os.path.expanduser(DEFAULT_ROOT)
        check_exists()
        os.chdir(os.path.join(os.path.expanduser(''), DEFAULT_ROOT))
        function(*args, **kwargs)
        os.chdir(root)
        return function(*args, **kwargs)
    return func


env_wrapper = partial(pseudo_wrapper, argument=arguments)
