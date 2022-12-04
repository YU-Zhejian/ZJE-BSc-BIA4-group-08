"""
Utility functions and tools for pytorch.
"""
__all__ = (
    "DictBackedTorchDataSet",
    "Describe",
    "get_torch_device"
)

from typing import Dict, Tuple, Optional

import torch
import torch.utils.data as tud
from torch import nn

from BIA_G8 import get_lh
from BIA_G8.helper import ndarray_helper

_lh = get_lh(__name__)


class DictBackedTorchDataSet(tud.Dataset):
    _index: Dict[int, Tuple[torch.Tensor, torch.Tensor]]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._index[index]

    def __len__(self) -> int:
        return len(self._index)

    def __init__(self, index: Optional[Dict[int, Tuple[torch.Tensor, torch.Tensor]]]=None) -> None:
        if index is None:
            index = {}
        self._index = dict(index)


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
    """
    Get suitable torch device. Will use NVidia GPU if possible.
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
