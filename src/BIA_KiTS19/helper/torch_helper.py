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
from torchvision.transforms import functional as TF

from BIA_KiTS19 import get_lh
from BIA_KiTS19.helper import dataset_helper, ndarray_helper

_lh = get_lh(__name__)


class KiTS19DataSet(tud.Dataset):
    """
    The KiTS19 dataset in a `pytorch` manner.
    """

    _dataset: Dict[int, dataset_helper.ImageSet]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_set = self._dataset[index]
        return (
            torch.unsqueeze(torch.tensor(image_set.np_image_final), dim=0),
            torch.unsqueeze(torch.tensor(image_set.np_mask_final), dim=0)
        )

    def __init__(self, dataset: dataset_helper.DataSet):
        self._dataset = {i: v for i, v in enumerate(dataset)}

    def __len__(self) -> int:
        return len(self._dataset)


class KiTS19DataSet2D(tud.Dataset):
    _index: Dict[int, Tuple[torch.Tensor, torch.Tensor]]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._index[index]

    def __init__(self, dataset: dataset_helper.DataSet, axis: int = 0):
        i = 0
        self._index = {}
        image_set: dataset_helper.ImageSet
        for image_set in tqdm.tqdm(iterable=dataset, desc="Loading Data..."):
            if image_set.np_image_final.shape != image_set.np_mask_final.shape:
                _lh.error(
                    f"image %s and mask %s shape mismatch!",
                    str(image_set.np_image_final.shape),
                    str(image_set.np_mask_final.shape)
                )
                continue
            for image_2d, mask_2d in zip(
                    ndarray_helper.sample_along_np(image_set.np_image_final, axis=axis),
                    ndarray_helper.sample_along_np(image_set.np_mask_final, axis=axis),
            ):
                self._index[i] = (
                    TF.to_tensor(image_2d),
                    torch.unsqueeze(torch.tensor(mask_2d), 0)
                )
                i += 1
                image_set.clear_all_cache()

    def __len__(self) -> int:
        return len(self._index)
