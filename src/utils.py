import os
from functools import wraps


def check_exists():
    return os.path.isdir(
        os.path.join(
            os.path.expanduser(''),
            'motion_dataset',
        )
    )


def change_to(path):
    def decorator(function):
        @wraps(function)
        def func(*args, **kwargs):
            current_path = os.getcwd()
            check_exists()
            os.chdir(os.path.join(os.path.expanduser(''), path))
            returns = function(*args, **kwargs)
            os.chdir(current_path)
            return returns
        return func
    return decorator
