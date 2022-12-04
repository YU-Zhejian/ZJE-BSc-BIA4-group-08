from __future__ import annotations

import json
from abc import abstractmethod
from typing import Final, Dict, Callable, Any, Iterable, Union, Type, Tuple, List

import numpy as np
import skimage.exposure as skiexp
import skimage.filters as skifilt
import skimage.filters.rank as skifiltrank
import skimage.restoration as skiresort
import skimage.transform as skitrans
from numpy import typing as npt

from BIA_G8 import get_lh
from BIA_G8.helper.io_helper import AbstractTOMLSerializable

_lh = get_lh(__name__)


class _Unset:
    pass


_unset = _Unset()


def _argument_string_to_int(instr: str) -> Union[_Unset, int]:
    return _unset if instr == "" else int(instr)


def _argument_string_to_float(instr: str) -> Union[_Unset, float]:
    return _unset if instr == "" else float(instr)


class LackRequiredArgumentError(ValueError):
    """Argument Parser Exception for lack of required argument"""

    def __init__(self, argument_names: str):
        super().__init__(
            f"Lack required arguments: {argument_names}"
        )


def _documentation_decorator(cls: Type[AbstractPreprocessor]) -> Type[AbstractPreprocessor]:
    """This class decorator generates documentations for arguments"""
    cls.__doc__ = f"{cls.__name__} ({cls._name}) -- {cls._description}\n\n"
    if not cls._arguments:
        cls.__doc__ += "No arguments available\n"
    else:
        for argument in cls._arguments:
            cls.__doc__ += f"* Argument ``{argument}`` \n"
    return cls


class AbstractPreprocessor(AbstractTOMLSerializable):
    """
    The abstraction of a general purposed preprocessing step.
    """
    _arguments: Dict[str, Callable[[str], Any]]
    _required_argument_names: List[str]
    _parsed_kwargs: Dict[str, Any]
    _description: str = "NOT AVAILABLE"
    _name: str = "UNNAMED"

    @abstractmethod
    def _function(self, img: npt.NDArray, **kwargs) -> npt.NDArray:
        pass

    @property
    def description(self) -> str:
        """Description of this preprocessor"""
        return self._description

    @property
    def name(self) -> str:
        """Name of this preprocessor"""
        return self._name

    @property
    def argument_names(self) -> Iterable[str]:
        return iter(self._arguments.keys())

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
        for argument_name, argument_value in kwargs.items():
            if argument_name not in self._arguments:
                continue
            parsed_argument_value = self._arguments[argument_name](argument_value)
            if parsed_argument_value is not _unset:
                self._parsed_kwargs[argument_name] = parsed_argument_value
        _lh.debug(repr(self))
        for argument_name in self._required_argument_names:
            if argument_name not in self._parsed_kwargs:
                raise LackRequiredArgumentError(argument_name)
        return self

    def execute(self, img: npt.NDArray) -> npt.NDArray:
        """
        Execute the preprocessor, which converts an image to another.

        :param img: Image to be preprocessed
        :return: Processed image
        """
        return self._function(img, **self._parsed_kwargs)

    def to_dict(self) -> Dict[str, Any]:
        return dict(self._parsed_kwargs)

    @classmethod
    def from_dict(cls, exported_dict: Dict[str, Any]) -> AbstractPreprocessor:
        return cls().set_params(**exported_dict)


@_documentation_decorator
class DumbPreprocessor(AbstractPreprocessor):
    _arguments: Final[Dict[str, Callable[[str], Any]]] = {}
    _required_argument_names: Final[List[str]] = []
    _name: Final[str] = "dumb"
    _description: Final[str] = "This preprocessor does nothing!"

    def _function(self, img: npt.NDArray, **kwargs) -> npt.NDArray:
        return img


@_documentation_decorator
class DimensionReductionPreprocessor(AbstractPreprocessor):
    _arguments: Final[Dict[str, Callable[[str], Any]]] = {}
    _required_argument_names: Final[List[str]] = []
    _name: Final[str] = "dimension reduction"

    def _function(self, img: npt.NDArray, **kwargs) -> npt.NDArray:
        if len(img.shape) == 3:
            img = img[:, :, 0]
        return skitrans.resize(img, (256, 256))


@_documentation_decorator
class AdjustExposurePreprocessor(AbstractPreprocessor):
    _arguments: Final[Dict[str, Callable[[str], Any]]] = {}
    _required_argument_names: Final[List[str]] = []
    _name: Final[str] = "adjust exposure"

    def _function(self, img: npt.NDArray, **kwargs) -> npt.NDArray:
        q2, q98 = np.percentile(img, (2, 98))
        img = skiexp.rescale_intensity(img, in_range=(q2, q98))
        img = skiexp.equalize_adapthist(img)
        return img


@_documentation_decorator
class DenoiseMedianPreprocessor(AbstractPreprocessor):
    _arguments: Final[Dict[str, Callable[[str], Any]]] = {
        "footprint_length_width": _argument_string_to_int
    }
    _required_argument_names: Final[List[str]] = []
    _name: Final[str] = "denoise (median)"

    def _function(self, img: npt.NDArray, **kwargs) -> npt.NDArray:
        if "footprint_length_width" in kwargs:
            footprint_length_width = kwargs["footprint_length_width"]
            footprint = np.ones(footprint_length_width, footprint_length_width)
            return skifiltrank.median(img, footprint=footprint)
        else:
            return skifiltrank.median(img)


@_documentation_decorator
class DenoiseMeanPreprocessor(AbstractPreprocessor):
    _arguments: Final[Dict[str, Callable[[str], Any]]] = {
        "footprint_length_width": _argument_string_to_int
    }
    _required_argument_names: Final[List[str]] = ["footprint_length_width"]
    _name: Final[str] = "denoise (mean)"

    def _function(self, img: npt.NDArray, **kwargs) -> npt.NDArray:
        footprint_length_width = kwargs["footprint_length_width"]
        footprint = np.ones(shape=(footprint_length_width, footprint_length_width))
        return skifiltrank.mean(img, footprint=footprint)


@_documentation_decorator
class DenoiseGaussianPreprocessor(AbstractPreprocessor):
    _arguments: Final[Dict[str, Callable[[str], Any]]] = {
        "sigma": _argument_string_to_int
    }
    _required_argument_names: Final[List[str]] = []
    _name: Final[str] = "denoise (gaussian)"

    def _function(self, img: npt.NDArray, **kwargs) -> npt.NDArray:
        return skifilt.gaussian(img, **kwargs)


@_documentation_decorator
class UnsharpMaskPreprocessor(AbstractPreprocessor):
    _arguments: Final[Dict[str, Callable[[str], Any]]] = {
        "radius": _argument_string_to_float,
        "amount": _argument_string_to_float,
    }
    _required_argument_names: Final[List[str]] = []
    _name: Final[str] = "unsharp mask"

    def _function(self, img: npt.NDArray, **kwargs) -> npt.NDArray:
        return skifilt.unsharp_mask(img, **kwargs)


@_documentation_decorator
class WienerDeblurPreprocessor(AbstractPreprocessor):
    _arguments: Final[Dict[str, Callable[[str], Any]]] = {
        "kernel_size": _argument_string_to_int,
        "balance": _argument_string_to_float,
    }
    _required_argument_names: Final[List[str]] = ["kernel_size", "balance"]
    _name: Final[str] = "wiener deblur"

    def _function(self, img: npt.NDArray, **kwargs) -> npt.NDArray:
        kernel = np.ones(kwargs["kernel_size"])
        balance = kwargs["balance"]
        return skiresort.wiener(img, psf=kernel / np.size(kernel), balance=balance)


_preprocessors: Dict[str, Type[AbstractPreprocessor]] = {
    cls._name: cls for cls in (
        DumbPreprocessor,
        DimensionReductionPreprocessor,
        AdjustExposurePreprocessor,
        DenoiseMeanPreprocessor,
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
