from __future__ import annotations

from typing import Any, Mapping

import numpy.typing as npt
import tomli
import tomli_w
from sklearn.neighbors import KNeighborsClassifier

from BIA_G8.helper import io_helper
from BIA_G8.model import AbstractClassifier


class SklearnKNearestNeighborsClassifier(AbstractClassifier):
    _name = "SklearnKNearestNeighborsClassifier"
    _knn: KNeighborsClassifier
    _hyper_params: Mapping[str, Any]

    def __init__(self, **hyper_params) -> None:
        self._hyper_params = hyper_params
        self._knn = KNeighborsClassifier(**hyper_params)

    def fit(self, data: npt.NDArray, label: npt.NDArray) -> SklearnKNearestNeighborsClassifier:
        self._knn.fit(X=data, y=label)
        return self

    def predict(self, data: npt.NDArray) -> npt.NDArray:
        return self._knn.predict(data)

    @classmethod
    def load(cls, model_abspath: str) -> SklearnKNearestNeighborsClassifier:
        hyper_params_path = model_abspath + ".hyper_params.toml"
        params = io_helper.read_pickle_xz(model_abspath)
        with open(hyper_params_path, "rb") as reader:
            hyper_params = tomli.load(reader)
        new_instance = cls(**hyper_params)
        new_instance._knn.set_params(**params)
        return new_instance

    def save(self, model_abspath: str) -> None:
        hyper_params_path = model_abspath + ".hyper_params.toml"
        params = self._knn.get_params()
        hyper_params = self._hyper_params
        io_helper.write_pickle_xz(params, model_abspath)
        with open(hyper_params_path, "wb") as writer:
            tomli_w.dump(hyper_params, writer)
