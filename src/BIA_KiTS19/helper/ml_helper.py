from typing import Any, Callable

import numpy as np
from numpy import typing as npt

from BIA_KiTS19 import get_lh
from BIA_KiTS19.helper import dataset_helper
from BIA_KiTS19.helper import ndarray_helper

_lh = get_lh(__name__)


def dice(mask_1: npt.NDArray[Any], mask_2: npt.NDArray[Any]):
    """Get similarity between 2 masks"""
    return np.sum(mask_1 == mask_2) / np.product(mask_1.size)


def evaluate(
        predictor: Callable[[npt.NDArray[float]], npt.NDArray[float]],
        image_set: dataset_helper.ImageSet
) -> float:
    predicted_mask = predictor(image_set.np_image_final)
    _lh.info("Evaluating predicted=%s with actual=%s", ndarray_helper.describe(predicted_mask),
             ndarray_helper.describe(image_set.np_mask_final))
    return dice(predicted_mask, image_set.np_mask_final)


def convert_2d_predictor_to_3d_predictor(
        predictor_2d: Callable[[npt.NDArray[float]], npt.NDArray[float]],
        axis: int = 0
) -> Callable[[npt.NDArray[float]], npt.NDArray[float]]:
    def predictor(img: npt.NDArray[float]) -> npt.NDArray[float]:
        return ndarray_helper.merge_along_np(map(
            predictor_2d,
            ndarray_helper.sample_along_np(array=img, axis=axis)
        ), axis=axis)

    return predictor
