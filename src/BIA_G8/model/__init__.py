import platform
import sys
import time
from typing import Dict

import BIA_G8

_lh = BIA_G8.get_lh(__name__)
_unknown_version = "UNKNOWN"


class DumbVersionable:
    __version__ = _unknown_version


try:
    import joblib
except ImportError:
    joblib = DumbVersionable()

try:
    import numpy as np
except ImportError:
    np = DumbVersionable()

try:
    import skimage
except ImportError:
    skimage = DumbVersionable()

try:
    import sklearn
except ImportError:
    sklearn = DumbVersionable()

try:
    import tomli
except ImportError:
    tomli = DumbVersionable()

try:
    import tomli_w
except ImportError:
    tomli_w = DumbVersionable()

try:
    import xgboost
except ImportError:
    xgboost = DumbVersionable()

try:
    import matplotlib
except ImportError:
    matplotlib = DumbVersionable()

try:
    import torch
except ImportError:
    torch = DumbVersionable()

try:
    import keras
except ImportError:
    keras = DumbVersionable()


def dump_versions() -> Dict[str, str]:
    return {
        "BIA_G8": BIA_G8.__version__,
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "joblib": joblib.__version__,
        "xgboost": xgboost.__version__,
        "matplotlib": matplotlib.__version__,
        "skimage": skimage.__version__,
        "tomli": tomli.__version__,
        "tomli-w": tomli_w.__version__,
        "torch": torch.__version__,
        "keras": keras.__version__,
        "python": ".".join(map(str, sys.version_info[0:3]))
    }


def validate_versions(compile_time_version_dict: Dict[str, str]) -> None:
    compile_time_version_dict = dict(compile_time_version_dict)
    run_time_version_dict = dump_versions()
    for version_key, run_time_version_value in run_time_version_dict.items():
        try:
            compile_time_version_value = compile_time_version_dict.pop(version_key)
        except KeyError:
            compile_time_version_value = _unknown_version
        if compile_time_version_value != run_time_version_value:
            _lh.warning(
                "Package %s have different version information: Compile (%s) != Run (%s)",
                version_key, compile_time_version_value, run_time_version_value
            )
    for remaining_compile_time_version_key, remaining_compile_time_version_value in compile_time_version_dict.items():
        _lh.warning(
            "Package %s have different version information: Compile (%s) != RUn (%s)",
            remaining_compile_time_version_key, remaining_compile_time_version_value, _unknown_version
        )


def dump_metadata() -> Dict[str, str]:
    return {
        "time_gmt": time.asctime(time.gmtime()),
        "time_local": time.asctime(time.localtime()),
        "platform_uname": platform.uname()._asdict()
    }


if __name__ == '__main__':
    print(dump_versions())
    print(dump_metadata())
