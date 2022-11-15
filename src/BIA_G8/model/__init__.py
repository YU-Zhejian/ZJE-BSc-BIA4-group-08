from __future__ import annotations

from abc import abstractmethod

from typing import Any, Mapping, Type, TypeVar

import joblib
import numpy.typing as npt
import tomli
from sklearn.neighbors import KNeighborsClassifier

from BIA_G8.helper import io_helper


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

    @classmethod
    @abstractmethod
    def new(self, **hyper_params) -> None:
        pass


_SKLearnModelType = TypeVar("_SKLearnModelType")


class BaseSklearnClassifier(AbstractClassifier):
    _model: _SKLearnModelType
    _model_type: Type[_SKLearnModelType]

    @classmethod
    def new(cls, **hyper_params) -> BaseSklearnClassifier:
        new_instance = cls()
        new_instance._model = new_instance._model_type(**hyper_params)
        return new_instance

    def fit(self, data: npt.NDArray, label: npt.NDArray) -> BaseSklearnClassifier:
        self._model.fit(X=data, y=label)
        return self

    def predict(self, data: npt.NDArray) -> npt.NDArray:
        return self._model.predict(data)

    @classmethod
    def load(cls, model_abspath: str) -> BaseSklearnClassifier:
        new_instance = cls()
        new_instance._model = joblib.load(model_abspath)
        return new_instance

    def save(self, model_abspath: str) -> None:
        joblib.dump(self._model, model_abspath)
