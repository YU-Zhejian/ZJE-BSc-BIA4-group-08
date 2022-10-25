"""
Here provides compressed readers and writers for Numpy and Torch serialization formats,
which can significantly reduce disk size.

The compression algorithm would be Lempel-Ziv Markov Chain Algorithm (LZMA) version 2 used in
`7-Zip <https://www.7-zip.org>`_. The implementation is provided Python standard library :py:mod:`lzma`.

.. warning::
    Since Python's standard LZMA implementation is single-threaded, it might be extremely slow to compress large objects!
"""

__all__ = (
    "read_np_xz",
    "read_tensor_xz",
    "write_np_xz",
    "write_tensor_xz"
)

import lzma
from typing import Any, Union, Mapping

import numpy as np
import numpy.typing as npt
import torch

from BIA_G8 import get_lh

_lh = get_lh(__name__)


def read_np_xz(path: str) -> npt.NDArray[Any]:
    """Reader of compressed Numpy serialization format"""
    with lzma.open(path, "rb") as reader:
        return np.load(reader)


def read_tensor_xz(path: str) -> Union[torch.Tensor, Mapping[str, Any]]:
    """Reader of compressed Torch serialization format"""
    with lzma.open(path, "rb") as reader:
        return torch.load(reader)


def write_np_xz(array: npt.NDArray[Any], path: str) -> None:
    """Writer of compressed Numpy serialization format"""
    with lzma.open(path, "wb", preset=9) as writer:
        np.save(writer, array)


def write_tensor_xz(array: Union[torch.Tensor, Mapping[str, Any]], path: str) -> None:
    """Writer of compressed Torch serialization format"""
    with lzma.open(path, "wb", preset=9) as writer:
        torch.save(array, writer)
