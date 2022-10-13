import SimpleITK as sitk
import numpy as np
import torch
from PIL import Image
from numpy import typing as npt
from torchvision.transforms import functional as TF

from BIA_KiTS19.helper import ndarray_helper


def sitk_to_normalized_np(img: sitk.Image) -> npt.NDArray[float]:
    return np.array(ndarray_helper.scale_np_array(sitk.GetArrayViewFromImage(img)), dtype=np.float16)


def np_2d_to_tensor(image_2d: npt.NDArray[float]) -> torch.Tensor:
    return TF.to_tensor(Image.fromarray(image_2d, mode="1"))


def tensor_to_np_2d(tensor: torch.Tensor) -> npt.NDArray[float]:
    return np.array(TF.to_pil_image(tensor[0, 0, :, :]))
