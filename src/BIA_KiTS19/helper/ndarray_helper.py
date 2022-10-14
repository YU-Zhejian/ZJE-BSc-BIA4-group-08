__all__ = (
    "sample_along_np",
    "merge_along_np",
    "scale_np_array",
    "sample_along_tensor",
    "describe"
)

from typing import Any, Iterable, Union, Tuple

import numpy as np
import torch
from numpy import typing as npt


def sample_along_np(
        array: npt.NDArray[Any],
        axis: int = 0,
        start: int = 0,
        end: int = -1,
        step: int = 1
) -> npt.NDArray:
    """
    Sample an image alongside an axis.

    :param array: A 2D/3D greyscale/RGB/RGBA image
    :param axis: Axis to be sampled. `0` or `1` for 2D image, `0`, `1` or `2` for 3D image.
    :param start: Sample start.
    :param end: Sample end. Use `-1` to make it the end of the image.
    :param step: Sample step.
    :return: Sampled image in `npt.NDArray`, which can be considered an array of image strips/ slices.
    """
    if end == -1:
        end = array.shape[axis]
    return np.moveaxis(array.take(indices=np.arange(start, end, step), axis=axis), axis, 0)


def merge_along_np(
        arrays: Iterable[npt.NDArray[Any]],
        axis: int = 0,
) -> npt.NDArray[Any]:
    imgs_list = list(arrays)
    first_img = imgs_list[0]
    merged_array = np.ndarray(shape=(len(imgs_list), *first_img.shape), dtype=first_img.dtype)
    for i, img in enumerate(imgs_list):
        merged_array[i, ...] = img
    return np.moveaxis(merged_array, 0, axis)


def sample_along_tensor(
        array: torch.Tensor,
        axis: int = 0,
        start: int = 0,
        end: int = -1,
        step: int = 1
) -> torch.Tensor:
    if end == -1:
        end = array.shape[axis]
    return torch.moveaxis(array.index_select(index=torch.arange(start, end, step), dim=axis), axis, 0)


def merge_along_np(
        array: Iterable[npt.NDArray[Any]],
        axis: int = 0,
) -> npt.NDArray[Any]:
    imgs_list = list(array)
    first_img = imgs_list[0]
    merged_array = np.ndarray(shape=(len(imgs_list), *first_img.shape), dtype=first_img.dtype)
    for i, img in enumerate(imgs_list):
        merged_array[i, ...] = img
    return np.moveaxis(merged_array, 0, axis)


def scale_np_array(x: npt.NDArray[Union[int, float]], out_range: Tuple[int, int] = (0, 1)):
    domain = np.min(x), np.max(x)
    y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
    return y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2


def scale_torch_array(x: torch.Tensor, out_range: Tuple[int, int] = (0, 1)):
    domain = torch.min(x), torch.max(x)
    y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
    return y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2


def describe(array: Union[npt.NDArray[Union[float, int]], torch.Tensor]) -> str:
    _quantiles = list(map(lambda f: f"{f:.2f}", np.quantile(array, q=[0, 0.25, 0.5, 0.75, 1])))
    return f"{type(array).__name__}[{array.dtype}] with shape={array.shape}; quantiles={_quantiles}"
