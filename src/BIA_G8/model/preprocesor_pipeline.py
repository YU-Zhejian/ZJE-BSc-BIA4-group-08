from __future__ import annotations

from typing import List, Dict, Union, Any, Iterable

import numpy.typing as npt
import tomli
import tomli_w

from BIA_G8 import get_lh
from BIA_G8.model.preprocessor import AbstractPreprocessor, get_preprocessor

_lh = get_lh(__name__)


class PreprocessorPipeline:
    _steps: List[AbstractPreprocessor]

    def __init__(self):
        self._steps = []

    def add_step(self, preprocessor: AbstractPreprocessor) -> PreprocessorPipeline:
        self._steps.append(preprocessor)
        return self

    def to_dict(self) -> Dict[str, Dict[str, Union[str, Dict[str, Any]]]]:
        return {
            str(i): {"name": step_name_args.name, "args": step_name_args.to_dict()}
            for i, step_name_args in enumerate(self._steps)
        }

    @classmethod
    def from_dict(cls, in_dict: Dict[str, Dict[str, Union[str, Dict[str, Any]]]]) -> PreprocessorPipeline:
        pp = cls()
        for _, step_name_args in in_dict.items():
            step = get_preprocessor(step_name_args["name"]).from_dict(step_name_args["args"])
            pp = pp.add_step(step)
        return pp

    @classmethod
    def load(cls, filename: str) -> PreprocessorPipeline:
        with open(filename, "rb") as reader:
            return cls.from_dict(tomli.load(reader))

    def save(self, filename: str) -> None:
        with open(filename, 'wb') as writer:
            tomli_w.dump(self.to_dict(), writer)

    def execute(self, img: npt.NDArray) -> npt.NDArray:
        for step in self._steps:
            img = step.execute(img)
        return img

    @property
    def steps(self) -> Iterable[AbstractPreprocessor]:
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
