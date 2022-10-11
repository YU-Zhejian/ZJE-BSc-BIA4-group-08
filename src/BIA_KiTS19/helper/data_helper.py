import glob
import os
from typing import Dict, Tuple, Optional, Iterable

import SimpleITK as sitk
import numpy as np
import numpy.typing as npt

from BIA_KiTS19.helper.sitk_helper import resample_spacing


class ImageSet:
    """
    Lazy-Evaluated Image set
    """
    _raw_nifti_image: Optional[sitk.Image] = None
    _raw_nifti_mask: Optional[sitk.Image] = None
    _space_resampled_nifti_image: Optional[sitk.Image] = None
    _space_resampled_nifti_mask: Optional[sitk.Image] = None
    _space_resampled_np_image: Optional[npt.NDArray] = None
    _space_resampled_np_mask: Optional[npt.NDArray] = None
    image_dir:str

    def __init__(self, image_dir:str = ""):
        self.image_dir = image_dir

    def _ensure_raw(self, property:str, cache_filename):
        if self.__getattribute__(property) is None:
            cache_path = os.path.join(self.image_dir, cache_filename)
            self.__setattr__(property, sitk.ReadImage(cache_path))
        return self.__getattribute__(property)

    @property
    def raw_nifti_image(self) -> sitk.Image:
        return self._ensure_raw("_raw_nifti_image", "imaging.nii.gz")

    @property
    def raw_nifti_mask(self) -> sitk.Image:
        if self._raw_nifti_mask is None:
            _cache_filename = os.path.join(os.path.join(self.image_dir, "segmentation.nii.gz"))
            self._raw_nifti_mask = sitk.ReadImage(_cache_filename)
        return self._raw_nifti_mask

    @property
    def space_resampled_nifti_image(self) -> sitk.Image:
        if self._space_resampled_nifti_image is None:
            _cache_filename = os.path.join(self.image_dir, "imaging.nii.gz")
            if os.path.exists(_cache_filename):
                self._space_resampled_nifti_image = sitk.ReadImage(_cache_filename)
            else:
                self._space_resampled_nifti_image = resample_spacing(self.raw_nifti_image)
                sitk.WriteImage(self._space_resampled_nifti_image, _cache_filename)
        return self._space_resampled_nifti_image

    @property
    def space_resampled_nifti_mask(self) -> sitk.Image:
        if self._space_resampled_nifti_mask is None:
            _cache_filename = os.path.join(self.image_dir, "imaging.nii.gz")
            if os.path.exists(_cache_filename):
                self._space_resampled_nifti_mask = sitk.ReadImage(_cache_filename)
            else:
                self._space_resampled_nifti_mask = resample_spacing(self.raw_nifti_image)
                sitk.WriteImage(self._space_resampled_nifti_mask, _cache_filename)
        return self._space_resampled_nifti_mask


class DataSet:
    data_dir: str
    _case_cache:Dict[str, Optional[Tuple[sitk.Image, sitk.Image]]]

    def __init__(self, data_dir:str = ""):
        self.data_dir = data_dir
        self._case_cache = {}
        for name in glob.glob(os.path.join(data_dir, "case_00*", "")):
            self._case_cache[name] = None

    def __getitem__(self, index:str) -> Tuple[sitk.Image, sitk.Image]:
        return self._case_cache[index]

    def iter_case_names(self) -> Iterable[str]:
        return iter(self._case_cache)

    def __iter__(self) -> Iterable[Tuple[sitk.Image, sitk.Image]]:
        return self.

def load_nifti(sample_name:str) -> sitk.Image:
    ...
