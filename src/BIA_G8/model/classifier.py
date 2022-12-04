from __future__ import annotations

import os
from abc import abstractmethod
from typing import TypeVar, Type, Final, Optional

import joblib
import numpy as np
import tomli_w
from numpy import typing as npt
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from BIA_G8.model import LackingOptionalRequirementError

try:
    from xgboost.sklearn import XGBClassifier
except ImportError:
    XGBClassifier = None


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
    _model_type: Optional[Type[_SKLearnModelType]]

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


class SklearnKNearestNeighborsClassifier(BaseSklearnClassifier):
    _name: Final[str] = "KNN (sklearn)"
    _model_type: Final[_SKLearnModelType] = KNeighborsClassifier


class SklearnSupportingVectorMachineClassifier(BaseSklearnClassifier):
    _name: Final[str] = "SVM (sklearn)"
    _model_type: Final[_SKLearnModelType] = SVC


class SklearnRandomForestClassifier(BaseSklearnClassifier):
    _name: Final[str] = "Random Forest (sklearn)"
    _model_type: Final[_SKLearnModelType] = RandomForestClassifier


class SklearnExtraTreesClassifier(BaseSklearnClassifier):
    _name: Final[str] = "Extra Trees (sklearn)"
    _model_type: Final[_SKLearnModelType] = ExtraTreesClassifier


class SklearnVotingClassifier(BaseSklearnClassifier):
    _name: Final[str] = "Voting (KNN, SVM, Decision Tree) (sklearn)"
    _model_type: Final[_SKLearnModelType] = VotingClassifier

    @classmethod
    def new(cls, **hyper_params) -> BaseSklearnClassifier:
        new_instance = cls()
        new_instance._model = new_instance._model_type(estimators=[
            ('knn', KNeighborsClassifier()),
            ('svm', SVC()),
            ('dt', DecisionTreeClassifier())
        ])
        return new_instance


class XGBoostClassifier(BaseSklearnClassifier):
    _name: Final[str] = "Gradient Boosting Tree (XGBoost)"
    _model_type: Final[_SKLearnModelType] = XGBClassifier

    @classmethod
    def new(cls, **hyper_params) -> BaseSklearnClassifier:
        new_instance = cls()
        if new_instance._model_type is None:
            raise LackingOptionalRequirementError(
                name="XGBoost",
                conda_name="py-xgboost-gpu",
                conda_channel="conda-forge",
                pypi_name="xgboost",
                url="https://xgboost.readthedocs.io"
            )
        new_instance._model = new_instance._model_type(**hyper_params)
        return new_instance


if __name__ == '__main__':
    x, y = make_classification(
        n_samples=12000,
        n_features=7,
        n_classes=3,
        n_clusters_per_class=1
    )
    x_train, x_test, y_train, y_test = train_test_split(x, y)
    m = XGBoostClassifier.new()
    m.fit(x_train, y_train)
    m.save("tmp.pkl.xz")
    del m
    m2 = XGBoostClassifier.load("tmp.pkl.xz")
    print(np.sum(m2.predict(x_test) == y_test) * 100 / len(y_test))
    os.remove("tmp.pkl.xz")
