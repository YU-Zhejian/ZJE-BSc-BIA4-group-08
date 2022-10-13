"""
Utility functions and tools for pytorch.
"""
from typing import Dict, Tuple

import torch
import torch.utils.data as tud
import tqdm

from BIA_KiTS19.helper import dataset_helper, ndarray_helper, converter


class KiTS19DataSet(tud.Dataset):
    """
    The KiTS19 dataset in a `pytorch` manner.
    """

    _dataset: Dict[int, dataset_helper.ImageSet]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_set = self._dataset[index]
        return (
            torch.tensor(image_set.np_image_final),
            torch.tensor(image_set.np_mask_final)
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
        for image_set in tqdm.tqdm(iterable=dataset, desc="Loading Data..."):
            if image_set.np_image_final.shape != image_set.np_mask_final.shape:
                print(f"ERR: {i}")
                continue
            for image_2d, mask_2d in zip(
                    ndarray_helper.sample_along(image_set.np_image_final, axis=axis),
                    ndarray_helper.sample_along(image_set.np_mask_final, axis=axis),
            ):
                self._index[i] = (
                    converter.np_2d_to_tensor(image_2d),
                    converter.np_2d_to_tensor(mask_2d)
                )
                i += 1
                image_set.clear_all_cache()

    def __len__(self) -> int:
        return len(self._index)
