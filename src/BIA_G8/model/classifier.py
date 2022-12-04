from __future__ import annotations

import os
from abc import abstractmethod
from typing import TypeVar, Type, Final, Optional, Iterable, Dict, Any

import joblib
import numpy as np
import skimage.transform as skitrans
import tomli
import tomli_w
import torch
import torch.optim
import torch.utils.data as tud
from torch import nn
from numpy import typing as npt
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from BIA_G8.helper.io_helper import read_tensor_xz, write_tensor_xz
from BIA_G8.helper.ml_helper import MachinelearningDatasetInterface
from BIA_G8.model import LackingOptionalRequirementError
from BIA_G8_DATA_ANALYSIS.covid_dataset import generate_fake_classification_dataset

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
    def fit(self, dataset: MachinelearningDatasetInterface) -> AbstractClassifier:
        pass

    @abstractmethod
    def predict(self, image: npt.NDArray) -> int:
        pass

    @abstractmethod
    def predicts(self, images: Iterable[npt.NDArray]) -> npt.NDArray:
        pass

    @abstractmethod
    def evaluate(self, test_dataset: MachinelearningDatasetInterface) -> float:
        pass

    @classmethod
    @abstractmethod
    def new(cls, **hyper_params) -> AbstractClassifier:
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

    def fit(self, dataset: MachinelearningDatasetInterface) -> BaseSklearnClassifier:
        self._model.fit(*dataset.sklearn_dataset)
        return self

    def predict(self, image: npt.NDArray) -> npt.NDArray:
        return self._model.predict(image.reshape(1, -1))[0]

    def predicts(self, images: npt.NDArray) -> npt.NDArray:
        image = images[0]
        return self._model.predict(images.reshape(-1, image.ravel().shape[0]))

    def evaluate(self, test_dataset: MachinelearningDatasetInterface) -> float:
        x_test, y_test = test_dataset.sklearn_dataset
        y_pred = self._model.predict(x_test)
        return np.sum(y_pred == y_test) / len(y_test)

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


class TorchClassifier(AbstractClassifier):
    _model: torch.Module
    _epoch: int
    _batch_size: int
    _lr: float
    _model_type: Type[torch.Module]
    _other_params: Dict[str, Any]

    def save(self, model_abspath: str) -> None:
        write_tensor_xz(self._model, model_abspath)
        hyper_params = {
            "epoch": self._epoch,
            "batch_size": self._batch_size,
            "lr": self._lr,
        }
        hyper_params.update(self._other_params)
        with open(".".join((model_abspath, "hyper_params")), "wb") as writer:
            tomli_w.dump(hyper_params, writer)

    @classmethod
    def load(cls, model_abspath: str) -> TorchClassifier:
        with open(".".join((model_abspath, "hyper_params")), "rb") as reader:
            hyper_params = tomli.load(reader)

        new_instance = cls.new(**hyper_params)
        new_instance._model = read_tensor_xz(model_abspath)
        return new_instance

    def fit(self, dataset: MachinelearningDatasetInterface) -> TorchClassifier:
        train_data_loader = tud.DataLoader(dataset.torch_dataset, batch_size=self._batch_size, shuffle=True)
        loss_func = nn.CrossEntropyLoss()
        opt = torch.optim.Adam(
            self._model.parameters(),
            lr=self._lr
        )
        for epoch in range(self._epoch):
            for i, (x_train, y_train) in enumerate(train_data_loader):
                y_train_pred_prob = self._model(x_train)
                y_train_pred = torch.argmax(y_train_pred_prob, dim=-1)
                loss = loss_func(y_train_pred_prob, y_train)
                opt.zero_grad()
                loss.backward()
                opt.step()
                accu = (torch.sum(y_train_pred == y_train) * 100 / y_train.shape[0]).item()
                if i % 10 == 0:
                    print(f"Epoch {epoch} batch {i}: accuracy {accu:.2f}")
        return self

    def predict(self, image: npt.NDArray) -> npt.NDArray:
        pass

    def predicts(self, images: Iterable[npt.NDArray]) -> npt.NDArray:
        pass

    @classmethod
    def new(
            cls,
            batch_size: int,
            epoch: int,
            lr: float,
            **other_params
    ) -> TorchClassifier:
        new_instance = cls()
        new_instance._batch_size = batch_size
        new_instance._lr = lr
        new_instance._epoch = epoch
        new_instance._other_params = other_params
        new_instance._model = cls._model_type(**new_instance._other_params)
        return new_instance

    def evaluate(self, test_dataset: MachinelearningDatasetInterface) -> float:
        test_data_loader = tud.DataLoader(test_dataset.torch_dataset, batch_size=self._batch_size, shuffle=True)
        correct_count = 0
        for i, (x_test, y_test) in enumerate(test_data_loader):
            y_test_pred_prob = self._model(x_test)
            y_pred = torch.argmax(y_test_pred_prob, dim=-1)
            correct_count += torch.sum(y_pred == y_test)
        return (correct_count / len(test_dataset)).item()


class ToyCNNModule(nn.Module):
    """
    Copy-and-paste from <https://blog.csdn.net/qq_45588019/article/details/120935828>
    """

    def __init__(
            self,
            n_features: int,
            n_classes: int
    ):
        super(ToyCNNModule, self).__init__()
        ksp = {
            "kernel_size": 3,
            "stride": 2,
            "padding": 1
        }
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=1,
                out_channels=16,
                **ksp
            ),
            torch.nn.BatchNorm2d(num_features=16),
            torch.nn.ReLU()
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=16,
                out_channels=32,
                **ksp
            ),
            torch.nn.BatchNorm2d(num_features=32),
            torch.nn.ReLU()
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=32,
                out_channels=64,
                **ksp
            ),
            torch.nn.BatchNorm2d(num_features=64),
            torch.nn.ReLU()
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=2,
                stride=2,
                padding=0
            ),
            torch.nn.BatchNorm2d(num_features=64),
            torch.nn.ReLU()
        )
        self.mlp1 = torch.nn.Linear(
            in_features=n_features // 4,
            out_features=64
        )
        self.mlp2 = torch.nn.Linear(
            in_features=64,
            out_features=n_classes
        )

    def forward(self, x: torch.Tensor):
        batch_size = x.shape[0]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.mlp1(x.view(batch_size, -1))
        x = self.mlp2(x)
        return x


class ToyCNNClassifier(TorchClassifier):
    _model_type:Final[Type[nn.Module]] = ToyCNNModule


def load_classifier_type(name: str) -> Type[AbstractClassifier]:
    ...


def load_classifier(name: str, parameter_path: str) -> AbstractClassifier:
    ...


if __name__ == '__main__':
    # ds = generate_fake_classification_dataset(300)
    # ds_train, ds_test = ds.train_test_split()
    # m = XGBoostClassifier.new()
    # m.fit(ds_train)
    # m.save("tmp.pkl.xz")
    # del m
    # m2 = XGBoostClassifier.load("tmp.pkl.xz")
    # print(m2.evaluate(ds_test))
    # os.remove("tmp.pkl.xz")

    ds = generate_fake_classification_dataset(300).parallel_apply(
        lambda img: skitrans.resize(
            img,
            (64, 64)
        )
    )
    ds_train, ds_test = ds.train_test_split()
    m = ToyCNNClassifier.new(batch_size=16, epoch=20, lr=0.0001, n_features=64*64, n_classes=3)
    m.fit(ds_train)
    m.save("tmp.pkl.xz")
    del m
    m2 = ToyCNNClassifier.load("tmp.pkl.xz")
    print(m2.evaluate(ds_test))
    os.remove("tmp.pkl.xz")
