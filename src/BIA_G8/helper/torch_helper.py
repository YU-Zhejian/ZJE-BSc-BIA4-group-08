"""
Utility functions and tools for pytorch.
"""
__all__ = (
    "DictBackedTorchDataSet",
    "get_torch_device"
)

from typing import Dict, Tuple, Optional

import numpy as np
import numpy.typing as npt
import torch
import torch.utils.data as tud

from BIA_G8 import get_lh
from BIA_G8.helper import ndarray_helper

_lh = get_lh(__name__)


class DictBackedTorchDataSet(tud.Dataset):
    """Torch dataset with dictionary as backend."""
    _index: Dict[int, Tuple[torch.Tensor, torch.Tensor]]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._index[index]

    def __len__(self) -> int:
        return len(self._index)

    def __init__(self, index: Optional[Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = None) -> None:
        """
        :param index: The dictionary
        """
        if index is None:
            index = {}
        self._index = dict(index)


def get_torch_device() -> torch.device:
    """
    Get suitable torch device. Will use NVidia GPU if possible.
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def convert_np_image_to_torch_tensor(img: npt.NDArray) -> torch.Tensor:
    """
    Convert numpy image to torch tensor.

    :param img: 2D single channel image in numpy format.
    :return: Torch tensor normalized to ``[0, 1]``
    """
    return torch.tensor(
        data=np.expand_dims(
            ndarray_helper.scale_np_array(img),
            axis=0
        ),
        dtype=torch.float
    )
