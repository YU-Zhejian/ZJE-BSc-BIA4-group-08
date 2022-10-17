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
    with lzma.open(path, "rb") as reader:
        return np.load(reader)


def read_tensor_xz(path: str) -> Union[torch.Tensor, Mapping[str, Any]]:
    with lzma.open(path, "rb") as reader:
        return torch.load(reader)


def write_np_xz(array: npt.NDArray[Any], path: str):
    with lzma.open(path, "wb", preset=9) as writer:
        np.save(writer, array)


def write_tensor_xz(array: Union[torch.Tensor, Mapping[str, Any]], path: str):
    with lzma.open(path, "wb", preset=9) as writer:
        torch.save(array, writer)
