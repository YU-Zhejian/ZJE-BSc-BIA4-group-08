import lzma
from typing import Any

import numpy as np
import numpy.typing as npt
import torch


def read_np_xz(path: str) -> npt.NDArray[Any]:
    with lzma.open(path, "rb") as reader:
        return np.load(reader)


def read_tensor_xz(path: str) -> torch.Tensor:
    with lzma.open(path, "rb") as reader:
        return torch.load(reader)


def save_np_xz(array: npt.NDArray[Any], path: str):
    with lzma.open(path, "wb", preset=9) as writer:
        np.save(writer, array)


def save_tensor_xz(array: torch.Tensor, path: str):
    with lzma.open(path, "wb", preset=9) as writer:
        torch.save(array, writer)
