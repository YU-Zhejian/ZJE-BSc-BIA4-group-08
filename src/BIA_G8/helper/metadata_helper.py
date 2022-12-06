"""
Helper functions that generates metadata for compatibility and debug purposes.
"""

from __future__ import annotations

__all__ = (
    "dump_metadata",
    "dump_versions",
    "validate_versions"
)

import platform
import sys
import time
from typing import Dict, Any

import BIA_G8

_lh = BIA_G8.get_lh(__name__)

_unknown_version = "UNKNOWN"


class _DumbVersion:
    __version__ = _unknown_version


try:
    import joblib
except ImportError:
    joblib = _DumbVersion()

try:
    import numpy as np
except ImportError:
    np = _DumbVersion()

try:
    import skimage
except ImportError:
    skimage = _DumbVersion()

try:
    import sklearn
except ImportError:
    sklearn = _DumbVersion()

try:
    import tomli
except ImportError:
    tomli = _DumbVersion()

try:
    import tomli_w
except ImportError:
    tomli_w = _DumbVersion()

try:
    import xgboost
except ImportError:
    xgboost = _DumbVersion()

try:
    import matplotlib
except ImportError:
    matplotlib = _DumbVersion()

try:
    import torch
except ImportError:
    torch = _DumbVersion()


def dump_versions() -> Dict[str, str]:
    """
    Dump current runtime version to a dictionary.
    """
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
        "python": ".".join(map(str, sys.version_info[0:3]))
    }


def validate_versions(compile_time_version_dict: Dict[str, str]) -> None:
    """
    Validate version compatibility information. Version that is not strictly matched would be reported in logs.

    :param compile_time_version_dict: Compile time versions.
    """
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


def dump_metadata() -> Dict[str, Any]:
    """
    Dump metadata to a dictionary.
    """
    return {
        "time_gmt": time.asctime(time.gmtime()),
        "time_local": time.asctime(time.localtime()),
        "platform_uname": platform.uname()._asdict()
    }
