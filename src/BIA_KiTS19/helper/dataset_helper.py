"""
The DataSet Abstraction
"""

import glob
import os
from typing import Dict, Optional, Iterable, Callable, Any, TypeVar, Tuple

import SimpleITK as sitk
import numpy as np
import numpy.typing as npt
import skimage.transform as skitrans

from BIA_KiTS19.helper import sitk_helper, converter

_CacheType = TypeVar("_CacheType")


class AbstractCachedLazyEvaluatedLinkedChain:
    """
    A lazy-evaluated dataset with disk cache, with following perspectives:

    - The dataset have several properties.
    - Properties can be generated from disk cache, or from some other properties, or both.
    """

    def _ensure_from_cache_file(
            self,
            cache_property_name: str,
            cache_filename: str,
            cache_file_reader: Callable[[str], _CacheType]
    ) -> _CacheType:
        """
        Ensure the existence of some property, which can be generated from disk cache.

        :param cache_property_name: Name of property cache.
        :param cache_filename: Name of on-disk cache.
        :param cache_file_reader: Function that loads the file.
        :return: The desired property, initialized.
        """
        if self.__getattribute__(cache_property_name) is None:
            if os.path.exists(cache_filename):
                self.__setattr__(cache_property_name, cache_file_reader(cache_filename))
            else:
                raise FileNotFoundError(f"Read desired property {cache_property_name} from {cache_filename} failed")
        return self.__getattribute__(cache_property_name)

    def _ensure_from_previous_step(
            self,
            cache_property_name: str,
            prev_step_property_name: str,
            transform_from_previous_step: Callable[[Any], _CacheType],

    ) -> _CacheType:
        """
        Ensure the existence of some property, which can be generated from disk cache or from a previous step.

        :param cache_property_name: Name of property cache.
        :param prev_step_property_name: Property of previous step.
        :param transform_from_previous_step: Transformer function that transforms data from previous step to this step.
        :return: The desired property, initialized.
        """
        if self.__getattribute__(cache_property_name) is None:
            self.__setattr__(
                cache_property_name,
                transform_from_previous_step(self.__getattribute__(prev_step_property_name))
            )
        return self.__getattribute__(cache_property_name)

    def _ensure_from_cache_and_previous_step(
            self,
            cache_property_name: str,
            cache_filename: str,
            cache_file_reader: Callable[[str], _CacheType],
            cache_file_writer: Callable[[_CacheType, str], None],
            prev_step_property_name: str,
            transform_from_previous_step: Callable[[Any], _CacheType],

    ) -> _CacheType:
        """
        Ensure the existence of some property, which can be generated from disk cache or from a previous step.

        :param cache_property_name: Name of property cache.
        :param cache_filename: Name of on-disk cache.
        :param cache_file_reader: Function that loads the file.
        :param cache_file_writer: Function that writes data to on-disk cache.
        :param prev_step_property_name: Property of previous step.
        :param transform_from_previous_step: Transformer function that transforms data from previous step to this step.
        :return: The desired property, initialized.
        """
        try:
            _property = self._ensure_from_cache_file(
                cache_property_name,
                cache_filename,
                cache_file_reader
            )
        except FileNotFoundError:
            _property = self._ensure_from_previous_step(
                cache_property_name,
                prev_step_property_name,
                transform_from_previous_step
            )
            cache_file_writer(_property, cache_filename)
        return _property


class ImageSet(AbstractCachedLazyEvaluatedLinkedChain):
    """
    Lazy-Evaluated Image set, which contains an image-mask pair in multiple formats.
    """
    _raw_nifti_image: Optional[sitk.Image]
    _raw_nifti_mask: Optional[sitk.Image]
    _space_resampled_nifti_image: Optional[sitk.Image]
    _space_resampled_nifti_mask: Optional[sitk.Image]
    _space_resampled_np_image: Optional[npt.NDArray[float]]
    _space_resampled_np_mask: Optional[npt.NDArray[float]]
    _rescaled_np_image: Optional[npt.NDArray[float]]
    _rescaled_np_mask: Optional[npt.NDArray[float]]
    image_dir: str
    """Directory where cases are saved"""

    def __init__(self, image_dir: str = ""):
        """
        :param image_dir: Directory where cases are saved
        """
        self.image_dir = image_dir
        self.clear_all_cache()

    @property
    def raw_nifti_image(self) -> sitk.Image:
        """The unprocessed raw nifti image in 3D ``sitk.Image``"""
        return self._ensure_from_cache_file(
            cache_property_name="_raw_nifti_image",
            cache_filename=os.path.join(self.image_dir, "imaging.nii.gz"),
            cache_file_reader=sitk.ReadImage
        )

    @property
    def raw_nifti_mask(self) -> sitk.Image:
        """The unprocessed raw nifti mask in 3D ``sitk.Image``"""
        return self._ensure_from_cache_file(
            cache_property_name="_raw_nifti_mask",
            cache_filename=os.path.join(self.image_dir, "segmentation.nii.gz"),
            cache_file_reader=sitk.ReadImage
        )

    @property
    def space_resampled_nifti_image(self) -> sitk.Image:
        """The nifti image in ``sitk.Image``, whose spacing was sampled to 1mm*1mm*1mm."""
        return self._ensure_from_cache_and_previous_step(
            cache_property_name="_space_resampled_nifti_image",
            cache_filename=os.path.join(self.image_dir, "imaging_space_resampled.nii.gz"),
            cache_file_reader=sitk.ReadImage,
            cache_file_writer=sitk.WriteImage,
            prev_step_property_name="raw_nifti_image",
            transform_from_previous_step=sitk_helper.resample_spacing
        )

    @property
    def space_resampled_nifti_mask(self) -> sitk.Image:
        """The nifti mask in ``sitk.Image``, whose spacing was sampled to 1mm*1mm*1mm."""
        return self._ensure_from_cache_and_previous_step(
            cache_property_name="_space_resampled_nifti_mask",
            cache_filename=os.path.join(self.image_dir, "mask_space_resampled.nii.gz"),
            cache_file_reader=sitk.ReadImage,
            cache_file_writer=sitk.WriteImage,
            prev_step_property_name="raw_nifti_mask",
            transform_from_previous_step=sitk_helper.resample_spacing
        )
    @property
    def space_resampled_np_image(self) -> npt.NDArray[float]:
        """The image in ``npt.NDArray.``, whose spacing was sampled to 1mm*1mm*1mm."""
        return self._ensure_from_cache_and_previous_step(
            cache_property_name="_space_resampled_np_image",
            cache_filename=os.path.join(self.image_dir, "imaging_space_resampled.npy"),
            cache_file_reader=np.load,
            cache_file_writer=lambda data, file_name: np.save(file_name, data),
            prev_step_property_name="space_resampled_nifti_image",
            transform_from_previous_step=converter.sitk_to_normalized_np
        )

    @property
    def space_resampled_np_mask(self) -> npt.NDArray[float]:
        """The mask in ``npt.NDArray.``, whose spacing was sampled to 1mm*1mm*1mm."""
        return self._ensure_from_cache_and_previous_step(
            cache_property_name="_space_resampled_np_mask",
            cache_filename=os.path.join(self.image_dir, "mask_space_resampled.npy"),
            cache_file_reader=np.load,
            cache_file_writer=lambda data, file_name: np.save(file_name, data),
            prev_step_property_name="space_resampled_nifti_mask",
            transform_from_previous_step=converter.sitk_to_normalized_np
        )

    @property
    def rescaled_np_image(self) -> npt.NDArray[float]:
        return self._ensure_from_cache_and_previous_step(
            cache_property_name="_rescaled_np_image",
            cache_filename=os.path.join(self.image_dir, "image_rescaled.npy"),
            cache_file_reader=np.load,
            cache_file_writer=lambda data, file_name: np.save(file_name, data),
            prev_step_property_name="space_resampled_np_image",
            transform_from_previous_step=lambda image: skitrans.resize(image, output_shape=(128,128,80))
        )

    @property
    def rescaled_np_mask(self) -> npt.NDArray[float]:
        return self._ensure_from_cache_and_previous_step(
            cache_property_name="_rescaled_np_mask",
            cache_filename=os.path.join(self.image_dir, "mask_rescaled.npy"),
            cache_file_reader=np.load,
            cache_file_writer=lambda data, file_name: np.save(file_name, data),
            prev_step_property_name="space_resampled_np_image",
            transform_from_previous_step=lambda image: skitrans.resize(image, output_shape=(128, 128, 80))
        )



    @property
    def np_image_final(self) -> npt.NDArray[float]:
        return self.rescaled_np_image

    @property
    def np_mask_final(self) -> npt.NDArray[float]:
        return self.rescaled_np_mask

    def clear_all_cache(self):
        """Clear all cache to save memory."""
        self._raw_nifti_image = None
        self._raw_nifti_mask = None
        self._space_resampled_nifti_image = None
        self._space_resampled_nifti_mask = None
        self._space_resampled_np_image = None
        self._space_resampled_np_mask = None
        self._rescaled_np_image = None
        self._rescaled_np_mask = None


class DataSet:
    data_dir: str
    _case_cache: Dict[str, ImageSet]

    def __init__(
            self,
            data_dir: str = "",
            range: Tuple[int, int] = (0, -1)
    ):
        self.data_dir = data_dir
        self._case_cache = {}
        for index, image_dir in enumerate(list(glob.glob(os.path.join(data_dir, "case_00*", "")))[range[0]:range[1]]):
            self._case_cache[image_dir] = ImageSet(image_dir=image_dir)

    def __getitem__(self, index: str) -> ImageSet:
        return self._case_cache[index]

    def iter_case_names(self) -> Iterable[str]:
        return iter(self._case_cache)

    def __iter__(self) -> Iterable[ImageSet]:
        return map(self.__getitem__, self.iter_case_names())

    def __len__(self):
        return len(self._case_cache)
