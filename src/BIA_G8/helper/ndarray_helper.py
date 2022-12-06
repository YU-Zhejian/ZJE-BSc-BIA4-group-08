"""
General-purposed helpers for Numpy NDArray and Torch Tensor.
"""

__all__ = (
    "scale_np_array",
    "scale_torch_array",
    "describe"
)

from typing import Union, Tuple, TypeVar

import numpy as np
import torch
from numpy import typing as npt

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
) -> npt.NDArray[float]:
    """
    Scale a Numpy array to specific range.

    See also: :py:func:`scale_torch_array`

    Example:

    >>> scale_np_array(np.array([1,2,3,4,5]), (0, 1))
    array([0.  , 0.25, 0.5 , 0.75, 1.  ])
    """
    domain = np.min(x), np.max(x)
    return _scale_impl(x, out_range, domain)


def scale_torch_array(
        x: torch.Tensor,
        out_range: Tuple[Union[int, float], Union[int, float]] = (0, 1)
) -> torch.Tensor:
    """
    Scale a Torch array to specific range.

    See also: :py:func:`scale_np_array`

    Example:

    >>> scale_torch_array(torch.tensor(np.array([1,2,3,4,5])), (0, 1))
    tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000])
    """
    domain = torch.min(x).item(), torch.max(x).item()
    return _scale_impl(x, out_range, domain)


def describe(array: _Tensor) -> str:
    """
    Describe the array by data type, shape, quantiles or unique values.

    Example:

    # FIXME: fails since int32 on Windows.

    >>> describe(np.array([0, 0, 1, 1]))
    'ndarray[int64] with shape=(4,); uniques=[0 1]'

    >>> describe(np.array(np.random.uniform(0, 21, size=12000), dtype=int))
    "ndarray[int64] with shape=(12000,); quantiles=['0.00', '5.00', '10.00', '15.00', '20.00']"

    :param array: The Numpy array or pyTorch Tensor to be described.
    :return: Description of the array.
    """
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
                _quantiles = list(map(lambda _q: f"{_q:.2f}", np.quantile(array, q=q)))
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
