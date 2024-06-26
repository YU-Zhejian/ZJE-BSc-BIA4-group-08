"""
Preprocessing pipeline abstraction
"""

from __future__ import annotations

from typing import List, Dict, Union, Any, Iterable

import numpy.typing as npt

from BIA_G8 import get_lh
from BIA_G8.helper.io_helper import AbstractTOMLSerializable
from BIA_G8.helper.ndarray_helper import scale_np_array
from BIA_G8.model.preprocessor import AbstractPreprocessor, get_preprocessor

_lh = get_lh(__name__)


class PreprocessorPipeline(AbstractTOMLSerializable):
    """
    A series of :py:mod:`AbstractPreprocessor` that can be written to or loaded from
    configuration file.
    """
    _steps: List[AbstractPreprocessor]

    def __init__(self):
        self._steps = []

    def add_step(self, preprocessor: AbstractPreprocessor) -> PreprocessorPipeline:
        """Append a preprocessor to current pipeline. Can be chained."""
        self._steps.append(preprocessor)
        return self

    def to_dict(self) -> Dict[str, Dict[str, Union[str, Dict[str, Any]]]]:
        retd = {
            str(i): {"name": step_name_args.name, "args": step_name_args.to_dict()}
            for i, step_name_args in enumerate(self._steps)
        }
        return retd

    @classmethod
    def from_dict(cls, in_dict: Dict[str, Dict[str, Union[str, Dict[str, Any]]]]):
        pp = cls()
        for _, step_name_args in in_dict.items():
            step = get_preprocessor(step_name_args["name"]).from_dict(step_name_args["args"])
            pp = pp.add_step(step)
        return pp

    def execute(self, img: npt.NDArray) -> npt.NDArray:
        """Execute the preprocessors sequentially over an image"""
        img = scale_np_array(img)
        for step in self._steps:
            img = step.execute(img)
        img = scale_np_array(img)
        return img

    @property
    def steps(self) -> Iterable[AbstractPreprocessor]:
        """Read-only preprocessors."""
        return iter(self._steps)

    def __eq__(self, other: PreprocessorPipeline) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return all(
            self_step == other_step
            for self_step, other_step in zip(
                self.steps,
                other.steps
            )
        )
