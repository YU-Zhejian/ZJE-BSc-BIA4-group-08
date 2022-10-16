import logging
import logging.handlers
import os.path
import sys

__all__ = (
    "__version__",
    "get_lh"
)

__version__ = "0.0.1"  # NOTE: Change ROOT_DIR/VERSION while updating

import time

sys.setrecursionlimit(int(1e8))
PACKAGE_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_DIR = os.path.dirname(
    os.path.dirname(PACKAGE_ROOT_DIR)
)

LOG_DIR = os.path.join(PROJECT_ROOT_DIR, "log")
os.makedirs(LOG_DIR, exist_ok=True)

streamFmt = logging.Formatter('[ %(levelname)-8s ] %(asctime)s.%(msecs)d %(name)s %(message)s')
fileFmt = logging.Formatter('[ %(levelname)-8s ] %(asctime)s:%(name)s %(filename)s:%(lineno)s %(message)s')

stream = logging.StreamHandler(sys.stderr)
stream.setLevel(logging.INFO)
stream.setFormatter(streamFmt)

log_file_path = os.path.join(LOG_DIR, f"{time.strftime('%Y_%m_%d-%H_%M_%S')}.log")
persistent_file = logging.FileHandler(log_file_path, 'w')
persistent_file.setLevel(logging.DEBUG)
persistent_file.setFormatter(fileFmt)


def get_lh(name: str = "ROOT") -> logging.Logger:
    """
    Get log handler.

    :param name: Name of the logger. Should be ``__name__`` of imported module.
    :return: generated logger.
    """
    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG)
    log.addHandler(stream)
    log.addHandler(persistent_file)
    return log


_lh = get_lh(__name__)
_lh.info("Logging set up successful at %s", log_file_path)
