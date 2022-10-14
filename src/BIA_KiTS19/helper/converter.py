__all__ = (
    "sitk_to_np",
    "sitk_to_normalized_np",
    "tensor_to_np",
    "mask_3d_to_4d",
    "mask_4d_to_3d"
)

from typing import TypeVar

import SimpleITK as sitk
import numpy as np
import torch
from numpy import typing as npt
from torchvision.transforms import functional as TF

from BIA_KiTS19.helper import ndarray_helper

_T = TypeVar("_T")


def sitk_to_np(img: sitk.Image, dtype: _T) -> npt.NDArray[_T]:
    return np.array(sitk.GetArrayViewFromImage(img), dtype=dtype)


def sitk_to_normalized_np(img: sitk.Image) -> npt.NDArray[float]:
    return np.array(ndarray_helper.scale_np_array(sitk.GetArrayViewFromImage(img)), dtype=np.float16)


def tensor_to_np(tensor: torch.Tensor) -> npt.NDArray[float]:
    return np.array(ndarray_helper.scale_np_array(np.array(TF.to_pil_image(tensor))), dtype=np.float16)


def mask_3d_to_4d(mask: npt.NDArray[int]) -> npt.NDArray[int]:
    colors = np.unique(mask.ravel())
    mask_4d = np.zeros(shape=(*mask.shape, len(colors)), dtype="uint8")
    for i in range(len(colors)):
        mask_4d[:, :, :, i] = np.array(1 * (mask == colors[i]), dtype=int)
    return mask_4d


def mask_4d_to_3d(mask: npt.NDArray[int]) -> npt.NDArray[float]:
    pass  # TODO
