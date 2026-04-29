import atexit
import os
import shutil
import signal
import sys
import tempfile
from typing import Any


def create_temp_file(prefix: str = "app-world-", suffix: str = ".tmp") -> str:
    return tempfile.mkstemp(prefix=prefix, suffix=suffix)[1]


def create_temp_directory(prefix: str = "app-world-") -> str:
    return tempfile.mkdtemp(prefix=prefix)


def clean_temp_directories(prefix: str = "app-world-") -> None:
    path = tempfile.mkdtemp()
    parent_path = os.path.dirname(path)
    for name in os.listdir(parent_path):
        path_ = os.path.join(parent_path, name)
        if not path_.startswith(prefix):
            continue
        if os.path.isfile(path_):
            os.remove(path_)
        if os.path.isdir(path_):
            shutil.rmtree(path_, ignore_errors=True)


def clean_temp_directories_on_exit(prefix: str = "app-world-") -> None:
    def signal_handler(signum: int, frame: Any) -> None:
        clean_temp_directories()
        sys.exit(0)

    atexit.register(clean_temp_directories)
    signal.signal(signal.SIGINT, signal_handler)
