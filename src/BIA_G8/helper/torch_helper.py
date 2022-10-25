"""
Utility functions and tools for pytorch.
"""
__all__ = (
)

from typing import Dict, Tuple

import torch
import torch.utils.data as tud

from BIA_G8 import get_lh

_lh = get_lh(__name__)


class AbstractTorchDataSet(tud.Dataset):
    _index: Dict[int, Tuple[torch.Tensor, torch.Tensor]]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._index[index]

    def __len__(self) -> int:
        return len(self._index)

    def __init__(self) -> None:
        self._index = {}


def get_torch_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
