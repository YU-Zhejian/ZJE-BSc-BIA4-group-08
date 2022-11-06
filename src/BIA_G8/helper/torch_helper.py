"""
Utility functions and tools for pytorch.
"""
__all__ = (
    "AbstractTorchDataSet",
    "Describe",
    "get_torch_device"
)

from typing import Dict, Tuple

import torch
import torch.utils.data as tud
from torch import nn

from BIA_G8 import get_lh
from BIA_G8.helper import ndarray_helper

_lh = get_lh(__name__)


class AbstractTorchDataSet(tud.Dataset):
    _index: Dict[int, Tuple[torch.Tensor, torch.Tensor]]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._index[index]

    def __len__(self) -> int:
        return len(self._index)

    def __init__(self) -> None:
        self._index = {}


class Describe(nn.Module):
    """
    The Describe Layer of PyTorch Module.

    Prints the description of matrix generated from last layer and pass the matrix without modification.
    """

    def __init__(self, prefix: str = ""):
        """
        The initializer

        :param prefix: Prefix of the printed message. Recommended to be the name of previous layer.

        See also: py:func:`BIA_G8.helper.ndarray_helper.describe`.
        """
        super().__init__()
        self.describe = lambda x: prefix + ndarray_helper.describe(x)

    def forward(self, x):
        """"""
        _lh.debug(self.describe(x))
        return x


def get_torch_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
