"""
Helpers related to SimpleITK (sitk) libraries for data preprocessing.
"""

__all__ = (
    "resample_spacing",
    "read_dicom"
)

import glob
import os.path
import warnings
from typing import Tuple, TypeVar

import SimpleITK as sitk
import numpy as np
from numpy import typing as npt

from BIA_G8.helper import ndarray_helper

warnings.warn("Module deprecated", DeprecationWarning)

_T = TypeVar("_T")


def resample_spacing(
        sitk_image: sitk.Image,
        new_space: Tuple[int, int, int] = (1, 1, 1)
) -> sitk.Image:
    img_size = sitk_image.GetSize()
    img_spacing = sitk_image.GetSpacing()
    origin = sitk_image.GetOrigin()
    direction = sitk_image.GetDirection()
    new_size = tuple(map(
        lambda i: int(img_size[i] * img_spacing[i] / new_space[i]),
        range(3)
    ))
    sitk_image = sitk.Resample(
        image1=sitk_image,
        size=new_size,
        transform=sitk.Euler3DTransform(),
        interpolator=sitk.sitkLinear,
        outputOrigin=origin,
        outputSpacing=new_space,
        outputDirection=direction
    )
    return sitk_image


def read_dicom(
        abs_path: str
) -> sitk.Image:
    """
    Read a DICOM file or directory of DICOM slices

    :param abs_path: Absolute path to DICOM file/directory
    :return: A SITK representation of image.
    """
    if os.path.isdir(abs_path):
        all_filenames = list(glob.glob(os.path.join(abs_path, "*.dcm")))
        if len(all_filenames) == 1:
            return read_dicom(all_filenames[0])
        elif len(all_filenames) == 0:
            raise FileNotFoundError(f"No valid DICOM file found at {abs_path}")
        else:
            reader = sitk.ImageSeriesReader()
            reader.MetaDataDictionaryArrayUpdateOn()
            reader.LoadPrivateTagsOn()
            series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(abs_path)
            dicom_names = reader.GetGDCMSeriesFileNames(abs_path, series_ids[0])
            reader.SetFileNames(dicom_names)
            return reader.Execute()
    else:
        return sitk.ReadImage(abs_path)


def sitk_to_np(img: sitk.Image, dtype: _T) -> npt.NDArray[_T]:
    return np.array(sitk.GetArrayViewFromImage(img), dtype=dtype)


def sitk_to_normalized_np(img: sitk.Image) -> npt.NDArray[float]:
    return np.array(ndarray_helper.scale_np_array(sitk.GetArrayViewFromImage(img)), dtype=np.float16)
