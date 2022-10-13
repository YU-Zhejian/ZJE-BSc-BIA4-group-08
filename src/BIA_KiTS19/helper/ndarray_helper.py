from typing import Any, Iterable, Union, Tuple

import numpy as np
from numpy import typing as npt

__all__ = (
    "sample_along",
    "merge_along",
    "scale_np_array"
)


def sample_along(
        img: npt.NDArray[Any],
        axis: int = 0,
        start: int = 0,
        end: int = -1,
        step: int = 1
) -> npt.NDArray:
    """
    Sample an image alongside an axis.

    :param img: A 2D/3D greyscale/RGB/RGBA image
    :param axis: Axis to be sampled. `0` or `1` for 2D image, `0`, `1` or `2` for 3D image.
    :param start: Sample start.
    :param end: Sample end. Use `-1` to make it the end of the image.
    :param step: Sample step.
    :return: Sampled image in `npt.NDArray`, which can be considered an array of image strips/ slices.
    """
    if end == -1:
        end = img.shape[axis]
    return np.moveaxis(img.take(indices=np.arange(start, end, step), axis=axis), axis, 0)


def merge_along(
        imgs: Iterable[npt.NDArray[Any]],
        axis: int = 0,
) -> npt.NDArray[Any]:
    imgs_list = list(imgs)
    first_img = imgs_list[0]
    merged_array = np.ndarray(shape=(len(imgs_list), *first_img.shape), dtype=first_img.dtype)
    for i, img in enumerate(imgs_list):
        merged_array[i, ...] = img
    return np.moveaxis(merged_array, 0, axis)


def scale_np_array(x: npt.NDArray[Union[int, float]], out_range: Tuple[int, int] = (0, 1)):
    domain = np.min(x), np.max(x)
    y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
    return y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2
