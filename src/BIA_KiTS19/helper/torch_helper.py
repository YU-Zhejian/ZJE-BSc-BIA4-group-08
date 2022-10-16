"""
Utility functions and tools for pytorch.
"""
__all__ = (
    "KiTS19DataSet",
    "KiTS19DataSet2D"
)

from typing import Dict, Tuple

import torch
import torch.utils.data as tud
import tqdm

from BIA_KiTS19 import get_lh
from BIA_KiTS19.helper import dataset_helper, ndarray_helper

_lh = get_lh(__name__)


class AbstractTorchDataSet(tud.Dataset):
    _index: Dict[int, Tuple[torch.Tensor, torch.Tensor]]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._index[index]

    def __len__(self) -> int:
        return len(self._index)

    def __init__(self):
        self._index = {}


class KiTS19DataSet(AbstractTorchDataSet):
    """
    The KiTS19 dataset in a `pytorch` manner.
    """

    def __init__(self, dataset: dataset_helper.DataSet):
        super().__init__()
        for index, image_set in tqdm.tqdm(iterable=enumerate(dataset), desc="Loading Data..."):
            image, mask = image_set.tensor_image_final, image_set.tensor_mask_final
            if image.shape != mask.shape:
                raise ndarray_helper.DimensionMismatchException(image, mask, "image", "mask")
            self._index[index] = (
                torch.unsqueeze(image, dim=0),
                torch.unsqueeze(mask, dim=0),
            )


class KiTS19DataSet2D(AbstractTorchDataSet):

    def __init__(self, dataset: dataset_helper.DataSet, axis: int = 0):
        super().__init__()
        index = 0
        image_set: dataset_helper.ImageSet
        for image_set in tqdm.tqdm(iterable=dataset, desc="Loading Data..."):
            image, mask = image_set.tensor_image_final, image_set.tensor_mask_final
            if image.shape != mask.shape:
                raise ndarray_helper.DimensionMismatchException(image, mask, "image", "mask")
            for image_2d, mask_2d in zip(
                    ndarray_helper.sample_along_tensor(image, axis=axis),
                    ndarray_helper.sample_along_tensor(mask, axis=axis),
            ):
                self._index[index] = (
                    torch.unsqueeze(image_2d, dim=0),
                    torch.unsqueeze(mask_2d, dim=0)
                )
                index += 1
                image_set.clear_all_cache()
