from __future__ import annotations

from abc import abstractmethod

import numpy.typing as npt


class AbstractClassifier:
    """
    Abstract Model for extension
    """

    _name: str

    @abstractmethod
    def save(self, model_abspath: str) -> None:
        pass

    @classmethod
    @abstractmethod
    def load(cls, model_abspath: str) -> AbstractClassifier:
        pass

    @abstractmethod
    def fit(self, data: npt.NDArray, label: npt.NDArray) -> AbstractClassifier:
        pass

    @abstractmethod
    def predict(self, data: npt.NDArray) -> npt.NDArray:
        pass

    @abstractmethod
    def __init__(self, **hyper_params) -> None:
        pass
