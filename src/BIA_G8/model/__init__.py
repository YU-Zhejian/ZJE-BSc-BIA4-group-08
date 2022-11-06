from __future__ import annotations

from abc import ABC, abstractmethod

import numpy.typing as npt


class AbstractModel(ABC):
    """
    Abstract Model for extension
    """

    _name: str

    @abstractmethod
    def save(self, model_abspath: str) -> None:
        pass

    @classmethod
    @abstractmethod
    def load(cls, model_abspath: str) -> AbstractModel:
        pass

    @abstractmethod
    def fit(self, data: npt.NDArray, label: npt.NDArray) -> AbstractModel:
        pass

    @abstractmethod
    def predict(self, data: npt.NDArray) -> npt.NDArray:
        pass
