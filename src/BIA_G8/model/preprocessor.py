from __future__ import annotations

import enum
import json
from abc import abstractmethod
from typing import Final, Dict, Any, Iterable, Type, Tuple, List

import numpy as np
import skimage.exposure as skiexp
import skimage.filters as skifilt
import skimage.filters.rank as skifiltrank
import skimage.restoration as skiresort
import skimage.transform as skitrans
from numpy import typing as npt

from BIA_G8 import get_lh
from BIA_G8.helper.io_helper import AbstractTOMLSerializable
from BIA_G8.helper.ndarray_helper import describe
from BIA_G8.model import unset, Argument, argument_string_to_int, argument_string_to_float, \
    LackRequiredArgumentError

_lh = get_lh(__name__)


def _documentation_decorator(cls: Type[AbstractPreprocessor]) -> Type[AbstractPreprocessor]:
    """This class decorator generates documentations for arguments"""
    cls.__doc__ = f"{cls.__name__} ({cls._name}) -- {cls._description}\n\n"
    if not cls._arguments:
        cls.__doc__ += "No arguments available\n"
    else:
        for argument in cls._arguments.values():
            cls.__doc__ += f"* {repr(argument)} \n"
    return cls


class ImageInputFormat(enum.IntEnum):
    ANY = 0
    UINT8 = 1
    FLOAT64_N1_1 = 2
    FLOAT64_0_1 = 3


class AbstractPreprocessor(AbstractTOMLSerializable):
    """
    The abstraction of a general purposed preprocessing step.
    """
    _arguments: Dict[str, Argument]
    _parsed_kwargs: Dict[str, Any]
    _description: str = "NOT AVAILABLE"
    _name: str = "UNNAMED"

    @abstractmethod
    def _function(self, img: npt.NDArray, **kwargs) -> npt.NDArray:
        raise NotImplementedError

    @property
    def description(self) -> str:
        """Description of this preprocessor"""
        return self._description

    @property
    def name(self) -> str:
        """Name of this preprocessor"""
        return self._name

    @property
    def arguments(self) -> Iterable[Argument]:
        return iter(self._arguments.values())

    def __init__(self) -> None:
        if not hasattr(self, "_arguments"):
            raise TypeError
        self._parsed_kwargs = {}

    def __repr__(self):
        return f"Initialized filter '{self._name}' with arguments {json.dumps(self._parsed_kwargs)}"

    def __eq__(self, other: AbstractPreprocessor) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self.to_dict() == other.to_dict()

    def set_params(self, **kwargs) -> AbstractPreprocessor:
        """
        Set arguments to the preprocessor. See the documentation of corresponding subclasses for more details.
        """
        for argument in self._arguments.values():
            if argument.name not in kwargs:
                if argument.is_required:
                    raise LackRequiredArgumentError(argument)
            parsed_argument_value = self._arguments[argument.name](kwargs[argument.name])
            if parsed_argument_value is not unset:
                self._parsed_kwargs[argument.name] = parsed_argument_value
            else:
                if argument.is_required:
                    raise LackRequiredArgumentError(argument)
        _lh.debug(repr(self))
        return self

    def execute(self, img: npt.NDArray) -> npt.NDArray:
        """
        Execute the preprocessor, which converts an image to another.

        :param img: Image to be preprocessed. Should be float64 in range [0, 1]
        :return: Processed image. Should be float64 in range [0, 1]
        """
        return self._function(img, **self._parsed_kwargs)

    def to_dict(self) -> Dict[str, Any]:
        return dict(self._parsed_kwargs)

    @classmethod
    def from_dict(cls, exported_dict: Dict[str, Any]) -> AbstractPreprocessor:
        return cls().set_params(**exported_dict)


@_documentation_decorator
class DumbPreprocessor(AbstractPreprocessor):
    _arguments: Final[Dict[str, Argument]] = {}
    _name: Final[str] = "dumb"
    _description: Final[str] = "This preprocessor does nothing!"

    def _function(self, img: npt.NDArray, **kwargs) -> npt.NDArray:
        return img


@_documentation_decorator
class DescribePreprocessor(AbstractPreprocessor):
    _arguments: Final[Dict[str, Argument]] = {}
    _name: Final[str] = "describe"
    _description: Final[str] = "This preprocessor describes status of current image"

    def _function(self, img: npt.NDArray, **kwargs) -> npt.NDArray:
        _lh.info(describe(img))
        return img


@_documentation_decorator
class NormalizationPreprocessor(AbstractPreprocessor):
    _arguments: Final[Dict[str, Argument]] = {}
    _name: Final[str] = "normalize"
    _description: Final[str] = "This preprocessor normalize the image for analysis"

    def _function(self, img: npt.NDArray, **kwargs) -> npt.NDArray:
        if len(img.shape) == 3:
            img = img[:, :, 0]
        img = skitrans.resize(img, (256, 256))
        return img


@_documentation_decorator
class AdjustExposurePreprocessor(AbstractPreprocessor):
    _arguments: Final[Dict[str, Argument]] = {}
    _name: Final[str] = "adjust exposure"
    _description: Final[str] = "This preprocessor can correct the image if it is underexposed or overexposed"

    def _function(self, img: npt.NDArray, **kwargs) -> npt.NDArray:
        q2, q98 = np.percentile(img, (2, 98))
        img = skiexp.rescale_intensity(img, in_range=(q2, q98))
        img = skiexp.equalize_adapthist(img)
        return img


@_documentation_decorator
class DenoiseMedianPreprocessor(AbstractPreprocessor):
    _arguments: Final[Dict[str, Argument]] = {
        argument.name: argument for argument in (
            Argument(
                name="footprint_length_width",
                description="degree of the filter",  # TODO
                is_required=False,
                parse_str=argument_string_to_int
            ),
        )
    }
    _name: Final[str] = "denoise (median)"
    _description: Final[str] = "This preprocessor can remove noise"

    def _function(self, img: npt.NDArray, **kwargs) -> npt.NDArray:
        if "footprint_length_width" in kwargs:
            footprint_length_width = kwargs["footprint_length_width"]
            footprint = np.ones(
                shape=(footprint_length_width, footprint_length_width),
                dtype=int
            )
            return skifiltrank.median(img, footprint=footprint)
        else:
            return skifiltrank.median(img)


@_documentation_decorator
class DenoiseMeanPreprocessor(AbstractPreprocessor):
    _arguments: Final[Dict[str, Argument]] = {
        argument.name: argument for argument in (
            Argument(
                name="footprint_length_width",
                description="",  # TODO
                is_required=True,
                parse_str=argument_string_to_int
            ),
        )
    }
    _name: Final[str] = "denoise (mean)"
    _description: Final[str] = "This preprocessor can remove noise"

    def _function(self, img: npt.NDArray, **kwargs) -> npt.NDArray:
        footprint_length_width = kwargs["footprint_length_width"]
        footprint = np.ones(
            shape=(footprint_length_width, footprint_length_width),
            dtype=int
        )
        return skifiltrank.mean(img, footprint=footprint)


@_documentation_decorator
class DenoiseGaussianPreprocessor(AbstractPreprocessor):
    _arguments: Final[Dict[str, Argument]] = {
        argument.name: argument for argument in (
            Argument(
                name="sigma",
                description="the degree of the blur effect of this filter",
                is_required=False,
                parse_str=argument_string_to_int
            ),
        )
    }
    _name: Final[str] = "denoise (gaussian)"
    _description: Final[str] = "This preprocessor can remove noise"

    def _function(self, img: npt.NDArray, **kwargs) -> npt.NDArray:
        return skifilt.gaussian(img, **kwargs)


@_documentation_decorator
class UnsharpMaskPreprocessor(AbstractPreprocessor):
    _arguments: Final[Dict[str, Argument]] = {
        argument.name: argument for argument in (
            Argument(
                name="radius",
                description="the larger it is, the more detail the image will loss",
                is_required=False,
                parse_str=argument_string_to_float
            ),
            Argument(
                name="amount",
                description="the extent that the image is enhanced",
                is_required=False,
                parse_str=argument_string_to_float
            ),
        )
    }
    _name: Final[str] = "unsharp mask"
    _description: Final[str] = "This preprocessor can sharp the images"

    def _function(self, img: npt.NDArray, **kwargs) -> npt.NDArray:
        return skifilt.unsharp_mask(img, **kwargs)


@_documentation_decorator
class WienerDeblurPreprocessor(AbstractPreprocessor):
    _arguments: Final[Dict[str, Argument]] = {
        argument.name: argument for argument in (
            Argument(
                name="kernel_size",
                description="the extent that the image is deblurred",  # TODO:
                is_required=True,
                parse_str=argument_string_to_int
            ),
            Argument(
                name="balance",
                description="",  # TODO:
                is_required=True,
                parse_str=argument_string_to_int
            ),
        )
    }
    _required_argument_names: Final[List[str]] = ["kernel_size", "balance"]
    _name: Final[str] = "wiener deblur"
    _description: Final[str] = "This preprocessor can deblur the images"

    def _function(self, img: npt.NDArray, **kwargs) -> npt.NDArray:
        kernel = np.ones(kwargs["kernel_size"])
        balance = kwargs["balance"]
        return skiresort.wiener(img, psf=kernel / np.size(kernel), balance=balance)


_preprocessors: Dict[str, Type[AbstractPreprocessor]] = {
    cls._name: cls for cls in (
        DumbPreprocessor,
        DescribePreprocessor,
        NormalizationPreprocessor,
        AdjustExposurePreprocessor,
        DenoiseMedianPreprocessor,
        DenoiseMeanPreprocessor,
        DenoiseGaussianPreprocessor,
        UnsharpMaskPreprocessor,
        WienerDeblurPreprocessor
    )
}


def get_preprocessor(preprocessor_name: str) -> Type[AbstractPreprocessor]:
    """Get :py:class:`AbstractPreprocessor` subclass using its name"""
    return _preprocessors[preprocessor_name]


def get_preprocessor_names() -> Iterable[str]:
    """Get a list of names of available :py:class:`AbstractPreprocessor` subclass"""
    return iter(_preprocessors.keys())


def get_preprocessor_name_descriptions() -> Iterable[Tuple[str, str]]:
    """
    Get a list of names and descriptions of available :py:class:`AbstractPreprocessor` subclass
    """
    return (
        (preprocessor_type().name, preprocessor_type().description)
        for preprocessor_type in _preprocessors.values()
    )
