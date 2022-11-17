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
            if not current_path == os.path.join(
                os.sep.join(current_path.split(os.sep)[:-2]),
                path,
            ):
                os.chdir(os.path.join(current_path, path))
                returns = function(*args, **kwargs)
                os.chdir(current_path)
            else:
                returns = function(*args, **kwargs)
            return returns
        return func
    return decorator
