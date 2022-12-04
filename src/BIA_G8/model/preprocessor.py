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

_lh = get_lh(__name__)


class _Unset:
    pass


_unset = _Unset()


def argument_string_to_int(instr: str) -> Union[_Unset, int]:
    return _unset if instr == "" else int(instr)


def argument_string_to_float(instr: str) -> Union[_Unset, float]:
    return _unset if instr == "" else float(instr)


class LackRequiredArgumentError(ValueError):
    def __init__(self, argument_names: str):
        super().__init__(
            f"Lack required arguments: {argument_names}"
        )


class AbstractPreprocessor:
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
        return self._description

    @property
    def name(self) -> str:
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

    def set_params(self, **kwargs) -> AbstractPreprocessor:
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
        return self._function(img, **self._parsed_kwargs)

    def to_dict(self) -> Dict[str, Any]:
        return dict(self._parsed_kwargs)

    @classmethod
    def from_dict(cls, exported_dict: Dict[str, Any]) -> AbstractPreprocessor:
        return cls().set_params(**exported_dict)


class DumbPreprocessor(AbstractPreprocessor):
    _arguments: Final[Dict[str, Callable[[str], Any]]] = {}
    _required_argument_names: Final[List[str]] = []
    _name: Final[str] = "dumb"
    _description: Final[str] = "This preprocessor does nothing!"

    def _function(self, img: npt.NDArray, **kwargs) -> npt.NDArray:
        return img


class DimensionReductionPreprocessor(AbstractPreprocessor):
    _arguments: Final[Dict[str, Callable[[str], Any]]] = {}
    _required_argument_names: Final[List[str]] = []
    _name: Final[str] = "dimension reduction"

    def _function(self, img: npt.NDArray, **kwargs) -> npt.NDArray:
        if len(img.shape) == 3:
            img = img[:, :, 0]
        return skitrans.resize(img, (256, 256))


class AdjustExposurePreprocessor(AbstractPreprocessor):
    _arguments: Final[Dict[str, Callable[[str], Any]]] = {}
    _required_argument_names: Final[List[str]] = []
    _name: Final[str] = "adjust exposure"

    def _function(self, img: npt.NDArray, **kwargs) -> npt.NDArray:
        q2, q98 = np.percentile(img, (2, 98))
        img = skiexp.rescale_intensity(img, in_range=(q2, q98))
        img = skiexp.equalize_adapthist(img)
        return img


class DenoiseMedianPreprocessor(AbstractPreprocessor):
    _arguments: Final[Dict[str, Callable[[str], Any]]] = {
        "footprint_length_width": argument_string_to_int
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


class DenoiseMeanPreprocessor(AbstractPreprocessor):
    _arguments: Final[Dict[str, Callable[[str], Any]]] = {
        "footprint_length_width": argument_string_to_int
    }
    _required_argument_names: Final[List[str]] = ["footprint_length_width"]
    _name: Final[str] = "denoise (mean)"

    def _function(self, img: npt.NDArray, **kwargs) -> npt.NDArray:
        footprint_length_width = kwargs["footprint_length_width"]
        footprint = np.ones(shape=(footprint_length_width, footprint_length_width))
        return skifiltrank.mean(img, footprint=footprint)


class DenoiseGaussianPreprocessor(AbstractPreprocessor):
    _arguments: Final[Dict[str, Callable[[str], Any]]] = {
        "sigma": argument_string_to_int
    }
    _required_argument_names: Final[List[str]] = []
    _name: Final[str] = "denoise (gaussian)"

    def _function(self, img: npt.NDArray, **kwargs) -> npt.NDArray:
        return skifilt.gaussian(img, **kwargs)


class UnsharpMaskPreprocessor(AbstractPreprocessor):
    _arguments: Final[Dict[str, Callable[[str], Any]]] = {
        "radius": argument_string_to_float,
        "amount": argument_string_to_float,
    }
    _required_argument_names: Final[List[str]] = []
    _name: Final[str] = "unsharp mask"

    def _function(self, img: npt.NDArray, **kwargs) -> npt.NDArray:
        return skifilt.unsharp_mask(img, **kwargs)


class WienerDeblurPreprocessor(AbstractPreprocessor):
    _arguments: Final[Dict[str, Callable[[str], Any]]] = {
        "kernel_size": argument_string_to_int,
        "balance": argument_string_to_float,
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
    return _preprocessors[preprocessor_name]


def get_preprocessor_names() -> Iterable[str]:
    return iter(_preprocessors.keys())


def get_preprocessor_name_descriptions() -> Iterable[Tuple[str, str]]:
    return (
        (preprocessor_type().name, preprocessor_type().description)
        for preprocessor_type in _preprocessors.values()
    )


if __name__ == "__main__":
    print(list(get_preprocessor_names()))
    assert get_preprocessor("dimension reduction")().execute(np.zeros(shape=(1024, 1024))).shape == (256, 256)
    dnm = get_preprocessor("denoise (mean)")()
    print(list(dnm.argument_names))
    dnm = dnm.set_params(footprint_length_width=5, aaa=6)
    assert dnm._parsed_kwargs == {'footprint_length_width': 5}
    assert dnm.execute(np.zeros(shape=(256, 256))).shape == (256, 256)
