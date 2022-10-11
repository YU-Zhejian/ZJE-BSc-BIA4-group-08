import glob
import os
from typing import Dict, Optional, Iterable, Callable, Any

import SimpleITK as sitk
import numpy as np
import numpy.typing as npt

from BIA_KiTS19.helper import sitk_helper


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
    image_dir: str

    def __init__(self, image_dir: str = ""):
        """
        :param image_dir: Directory where cases are saved
        """
        self.image_dir = image_dir

    def _ensure_raw(self, _property_name: str, cache_filename: str):
        if self.__getattribute__(_property_name) is None:
            cache_path = os.path.join(self.image_dir, cache_filename)
            self.__setattr__(_property_name, sitk.ReadImage(cache_path))
        return self.__getattribute__(_property_name)

    @property
    def raw_nifti_image(self) -> sitk.Image:
        return self._ensure_raw("_raw_nifti_image", "imaging.nii.gz")

    @property
    def raw_nifti_mask(self) -> sitk.Image:
        return self._ensure_raw("_raw_nifti_mask", "segmentation.nii.gz")

    def _ensure(
            self,
            cache_property_name: str,
            cache_filename: str,
            cache_file_loader: Callable[[str], Any],
            cache_file_writer: Callable[[Any, str], None],
            prev_step_property_name: str,
            transform_from_previous_step: Callable[[Any], Any],

    ):
        """
        Ensure the existence of some property, which can be generated from disk cache or from a previous step.

        :param cache_property_name: Name of property cache.
        :param cache_filename: Name of on-disk cache.
        :param cache_file_loader: Function that loads the file.
        :param cache_file_writer: Function that writes data to on-disk cache.
        :param prev_step_property_name: Property of previous step.
        :param transform_from_previous_step: Transformer function that transforms data from previous step to this step.
        :return: The desired property, initialized.
        """
        if self.__getattribute__(cache_property_name) is None:
            _cache_filename = os.path.join(self.image_dir, cache_filename)
            if os.path.exists(_cache_filename):
                self.__setattr__(cache_property_name, cache_file_loader(_cache_filename))
            else:
                self.__setattr__(cache_property_name,
                                 transform_from_previous_step(self.__getattribute__(prev_step_property_name)))
                cache_file_writer(self.__getattribute__(cache_property_name), _cache_filename)
        return self.__getattribute__(cache_property_name)

    @property
    def space_resampled_nifti_image(self) -> sitk.Image:
        return self._ensure(
            cache_property_name="_space_resampled_nifti_image",
            cache_filename="imaging_space_resampled.nii.gz",
            cache_file_loader=sitk.ReadImage,
            cache_file_writer=sitk.WriteImage,
            prev_step_property_name="raw_nifti_image",
            transform_from_previous_step=sitk_helper.resample_spacing
        )

    @property
    def space_resampled_nifti_mask(self) -> sitk.Image:
        return self._ensure(
            cache_property_name="_space_resampled_nifti_mask",
            cache_filename="mask_space_resampled.nii.gz",
            cache_file_loader=sitk.ReadImage,
            cache_file_writer=sitk.WriteImage,
            prev_step_property_name="raw_nifti_mask",
            transform_from_previous_step=sitk_helper.resample_spacing
        )

    @property
    def space_resampled_np_image(self) -> npt.NDArray:
        return self._ensure(
            cache_property_name="_space_resampled_np_image",
            cache_filename="imaging_space_resampled.npy",
            cache_file_loader=np.load,
            cache_file_writer=np.save,
            prev_step_property_name="space_resampled_nifti_image",
            transform_from_previous_step=sitk.GetArrayViewFromImage
        )

    @property
    def space_resampled_np_mask(self) -> npt.NDArray:
        return self._ensure(
            cache_property_name="_space_resampled_np_mask",
            cache_filename="mask_space_resampled.npy",
            cache_file_loader=np.load,
            cache_file_writer=np.save,
            prev_step_property_name="space_resampled_nifti_mask",
            transform_from_previous_step=sitk.GetArrayViewFromImage
        )


class DataSet:
    data_dir: str
    _case_cache: Dict[str, ImageSet]

    def __init__(self, data_dir: str = ""):
        self.data_dir = data_dir
        self._case_cache = {}
        for image_dir in glob.glob(os.path.join(data_dir, "case_00*", "")):
            self._case_cache[image_dir] = ImageSet(image_dir=image_dir)

    def __getitem__(self, index: str) -> ImageSet:
        return self._case_cache[index]

    def iter_case_names(self) -> Iterable[str]:
        return iter(self._case_cache)

    def __iter__(self) -> Iterable[ImageSet]:
        return map(self.__getitem__, self.iter_case_names())

    def __len__(self):
        return len(self._case_cache)
