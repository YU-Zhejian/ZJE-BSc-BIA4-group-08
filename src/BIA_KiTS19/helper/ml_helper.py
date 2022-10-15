__al__ = (
    "dice",
    "evaluate",
    "convert_2d_predictor_to_3d_predictor"
)

from typing import Any, Callable

import numpy as np
import torch
from numpy import typing as npt

from BIA_KiTS19 import get_lh
from BIA_KiTS19.helper import dataset_helper

_lh = get_lh(__name__)


def get_torch_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def dice(mask_1: npt.NDArray[Any], mask_2: npt.NDArray[Any]):
    """Get similarity between 2 masks"""
    return np.sum(mask_1 == mask_2) / np.product(mask_1.size)


def evaluate(
        predictor: Callable[[npt.NDArray[float]], npt.NDArray[float]],
        image_set: dataset_helper.ImageSet
) -> float:
    predicted_mask = predictor(image_set.np_image_final)
    # print(f"predicted_mask: {ndarray_helper.describe(predicted_mask)}")
    # print(f"image_set.np_mask_final: {ndarray_helper.describe(image_set.np_mask_final)}")
    return dice(predicted_mask, image_set.np_mask_final)
