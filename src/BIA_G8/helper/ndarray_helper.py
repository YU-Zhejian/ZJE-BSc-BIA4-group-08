__all__ = (
    "sample_along_np",
    "merge_along_np",
    "scale_np_array",
    "sample_along_tensor",
    "describe"
)

from typing import Any, Iterable, Union, Tuple, TypeVar

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


def merge_along_tensor(
        arrays: Iterable[torch.Tensor],
        axis: int = 0,
) -> torch.Tensor:
    imgs_list = list(arrays)
    first_img = imgs_list[0]
    merged_array = torch.zeros(size=(len(imgs_list), *first_img.shape), dtype=first_img.dtype)
    for i, img in enumerate(imgs_list):
        merged_array[i, ...] = img
    return torch.moveaxis(merged_array, 0, axis)


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


_Tensor = TypeVar("_Tensor", npt.NDArray[Union[float, int]], torch.Tensor)


def _scale_impl(
        x: _Tensor,
        out_range: Tuple[Union[int, float], Union[int, float]],
        domain: Tuple[Union[int, float], Union[int, float]]
) -> _Tensor:
    if domain[1] == domain[0]:
        return x
    y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
    return y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2


def scale_np_array(
        x: npt.NDArray[Union[int, float]],
        out_range: Tuple[Union[int, float], Union[int, float]] = (0, 1)
) -> npt.NDArray[Union[int, float]]:
    domain = np.min(x), np.max(x)
    return _scale_impl(x, out_range, domain)


def scale_torch_array(
        x: torch.Tensor,
        out_range: Tuple[Union[int, float], Union[int, float]] = (0, 1)
) -> torch.Tensor:
    domain = torch.min(x), torch.max(x)
    return _scale_impl(x, out_range, domain)


def describe(array: _Tensor) -> str:
    q = [0, 0.25, 0.5, 0.75, 1]
    _shape = tuple(array.shape)
    if isinstance(array, torch.Tensor):
        _unique = array.unique()
    else:
        _unique = np.unique(array)
    if len(_unique) > 10:
        try:
            if isinstance(array, torch.Tensor):
                array = array.float()
                _quantiles = list(map(lambda _q: f"{array.quantile(q=_q):.2f}", q))
            else:
                _quantiles = np.quantile(array, q=q)
        except (IndexError, RuntimeError) as e:
            _quantiles = f"ERROR {e}"
        _quantiles_str = f"quantiles={_quantiles}"
    else:
        _quantiles_str = f"uniques={_unique}"
    return f"{type(array).__name__}[{array.dtype}] with shape={_shape}; {_quantiles_str}"


class DimensionMismatchException(ValueError):
    def __init__(
            self,
            _arr1: _Tensor,
            _arr2: _Tensor,
            _arr1_name: str = "arr1",
            _arr2_name: str = "arr2"
    ):
        super().__init__(
            f"Array {_arr1_name} and {_arr2_name}  dimension mismatch!\n"
            f"\twhere {_arr1_name} is {describe(_arr1)}\n"
            f"\twhere {_arr2_name} is {describe(_arr2)}\n"
        )
