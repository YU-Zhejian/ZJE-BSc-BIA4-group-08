from __future__ import annotations

import os.path
from abc import abstractmethod
from typing import TypeVar, Type, Final, Optional, Iterable, Dict, Any

import joblib
import numpy as np
import torch
import torch.optim
import torch.utils.data as tud
from numpy import typing as npt
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from torch import nn

from BIA_G8 import get_lh
from BIA_G8.helper.io_helper import read_tensor_xz, write_tensor_xz, SerializableInterface, read_toml_with_metadata, \
    write_toml_with_metadata
from BIA_G8.helper.ml_helper import MachinelearningDatasetInterface
from BIA_G8.helper.torch_helper import AbstractTorchModule
from BIA_G8.model import LackingOptionalRequirementError

_lh = get_lh(__name__)

try:
    from xgboost.sklearn import XGBClassifier
except ImportError:
    XGBClassifier = None


class AbstractClassifier(SerializableInterface):
    """
    Abstract Model for extension
    """
    _name: str

    @property
    def name(self) -> str:
        return self._name

    @abstractmethod
    def fit(self, dataset: MachinelearningDatasetInterface) -> AbstractClassifier:
        raise NotImplementedError

    @abstractmethod
    def predict(self, image: npt.NDArray) -> int:
        raise NotImplementedError

    @abstractmethod
    def predicts(self, images: Iterable[npt.NDArray]) -> npt.NDArray:
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, test_dataset: MachinelearningDatasetInterface) -> float:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def new(cls, **params) -> AbstractClassifier:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(self, path: str, load_model: bool = True) -> AbstractClassifier:
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str, save_model: bool = True) -> None:
        raise NotImplementedError


_SKLearnModelType = TypeVar("_SKLearnModelType")


class BaseSklearnClassifier(AbstractClassifier):
    _model: _SKLearnModelType
    _model_type: Optional[Type[_SKLearnModelType]]
    _params: Dict[str, Any]

    def __init__(self, *, model: _SKLearnModelType, **params):
        self._params = params
        self._model = model

    @classmethod
    def new(cls, **params) -> BaseSklearnClassifier:
        return cls(model=cls._model_type(**params), **params)

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
    def load(cls, path: str, load_model: bool = True) -> BaseSklearnClassifier:
        loaded_data = read_toml_with_metadata(path)
        if "model_path" in loaded_data and load_model:
            _lh.info("%s: Loading pretrained model...", cls.__name__)
            return cls(model=joblib.load(loaded_data["model_path"]), **loaded_data["params"])
        else:
            _lh.info("%s: Loading parameters only...", cls.__name__)
            return cls.new(**loaded_data["params"])

    def save(self, path: str, save_model: bool = True) -> None:
        path = os.path.abspath(path)
        out_dict = {
            "name": self._name,
            "params": self._params
        }
        if save_model:
            model_path = path + ".pkl.xz"
            joblib.dump(self._model, model_path)
            out_dict["model_path"] = model_path
        write_toml_with_metadata(out_dict, path)


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
    def new(cls, **params) -> BaseSklearnClassifier:
        new_instance = cls(model=cls._model_type(estimators=[
            ('knn', KNeighborsClassifier()),
            ('svm', SVC()),
            ('dt', DecisionTreeClassifier())
        ]))
        return new_instance


class XGBoostClassifier(BaseSklearnClassifier):
    _name: Final[str] = "Gradient Boosting Tree (XGBoost)"
    _model_type: Final[_SKLearnModelType] = XGBClassifier

    @classmethod
    def new(cls, **params) -> BaseSklearnClassifier:
        if cls._model_type is None:
            raise LackingOptionalRequirementError(
                name="XGBoost",
                conda_name="py-xgboost-gpu",
                conda_channel="conda-forge",
                pypi_name="xgboost",
                url="https://xgboost.readthedocs.io"
            )
        new_instance = cls(model=cls._model_type(**params), **params)
        return new_instance


class BaseTorchClassifier(AbstractClassifier):
    _model: AbstractTorchModule
    _device: str
    _num_epochs: int
    _batch_size: int
    _lr: float
    _model_type: Type[AbstractTorchModule]
    _model_params: Dict[str, Any]
    _hyper_params: Dict[str, Any]

    def __init__(
            self,
            *,
            model: AbstractTorchModule,
            hyper_params: Dict[str, Any],
            model_params: Dict[str, Any]
    ):
        self._hyper_params = hyper_params
        self._batch_size = hyper_params["batch_size"]
        self._num_epochs = hyper_params["num_epochs"]
        self._device = hyper_params["device"]
        self._lr = hyper_params["lr"]
        self._model_params = model_params
        self._model = model

    @classmethod
    def load(cls, path: str, load_model: bool = True) -> BaseTorchClassifier:
        loaded_data = read_toml_with_metadata(path)
        if "model_path" in loaded_data and load_model:
            _lh.info("%s: Loading pretrained model...", cls.__name__)
            return cls(
                model=read_tensor_xz(loaded_data["model_path"]),
                hyper_params=loaded_data["hyper_params"],
                model_params=loaded_data["model_params"]
            )
        else:
            _lh.info("%s: Loading parameters only...", cls.__name__)
            return cls.new(hyper_params=loaded_data["hyper_params"], model_params=loaded_data["model_params"])

    def save(self, path: str, save_model: bool = True) -> None:
        path = os.path.abspath(path)
        out_dict = {
            "name": self._name,
            "hyper_params": self._hyper_params,
            "model_params": self._model_params
        }
        if save_model:
            model_path = path + ".pt.xz"
            write_tensor_xz(self._model, model_path)
            out_dict["model_path"] = model_path
        write_toml_with_metadata(out_dict, path)

    def fit(self, dataset: MachinelearningDatasetInterface) -> BaseTorchClassifier:
        self._model = self._model.to(self._device)
        train_data_loader = tud.DataLoader(dataset.torch_dataset, batch_size=self._batch_size, shuffle=True)
        loss_func = nn.CrossEntropyLoss()
        opt = torch.optim.Adam(
            self._model.parameters(),
            lr=self._lr
        )
        for epoch in range(self._num_epochs):
            for i, (x_train, y_train) in enumerate(train_data_loader):
                x_train, y_train = x_train.to(self._device), y_train.to(self._device)
                y_pred_prob = self._model(x_train)
                y_pred = torch.argmax(y_pred_prob, dim=-1)
                loss = loss_func(y_pred_prob, y_train)
                opt.zero_grad()
                loss.backward()
                opt.step()
                accu = (torch.sum(torch.eq(y_pred, y_train)) * 100 / y_train.shape[0]).item()
                if i % 10 == 0:
                    _lh.info(f"%s Epoch {epoch} batch {i}: accuracy {accu:.2f}", self.__class__.__name__)
        return self

    def predict(self, image: npt.NDArray) -> npt.NDArray:
        raise NotImplementedError

    def predicts(self, images: Iterable[npt.NDArray]) -> npt.NDArray:
        raise NotImplementedError

    @classmethod
    def new(
            cls,
            hyper_params: Dict[str, Any],
            model_params: Dict[str, Any]
    ) -> BaseTorchClassifier:
        return cls(
            hyper_params=hyper_params,
            model_params=model_params,
            model=cls._model_type(**model_params)
        )

    def evaluate(self, test_dataset: MachinelearningDatasetInterface) -> float:
        test_data_loader = tud.DataLoader(test_dataset.torch_dataset, batch_size=self._batch_size, shuffle=True)
        correct_count = 0
        for i, (x_test, y_test) in enumerate(test_data_loader):
            x_test, y_test = x_test.to(self._device), y_test.to(self._device)
            y_pred = torch.argmax(self._model(x_test), dim=-1)
            correct_count += torch.sum(torch.eq(y_pred, y_test)).item()
        return correct_count / len(test_dataset)


class ToyCNNModule(AbstractTorchModule):
    """
    Copy-and-paste from <https://blog.csdn.net/qq_45588019/article/details/120935828>
    """

    def __init__(
            self,
            n_features: int,
            n_classes: int,
            kernel_size: int,
            stride: int,
            padding: int
    ):
        super(ToyCNNModule, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            ),
            torch.nn.BatchNorm2d(num_features=16),
            torch.nn.ReLU()
        )
        """
        [BATCH_SIZE, 1, WID, HEIGHT] -> [BATCH_SIZE, 16, WID // 2, HEIGHT // 2]
        """
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            ),
            torch.nn.BatchNorm2d(num_features=32),
            torch.nn.ReLU()
        )
        """
        [BATCH_SIZE, 16, WID // 2, HEIGHT // 2] -> [BATCH_SIZE, 32, WID // 4, HEIGHT // 4]
        """
        self.mlp1 = torch.nn.Linear(
            in_features=32 * n_features // (4 ** 2),
            out_features=64
        )
        """
        [BATCH_SIZE, 32, WID // 4, HEIGHT // 4] -> [BATCH_SIZE, 64]
        """
        self.mlp2 = torch.nn.Linear(
            in_features=64,
            out_features=n_classes
        )
        """
        [BATCH_SIZE, 64] -> [BATCH_SIZE, 3]
        """
        # self.describe = Describe()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        # x = self.describe(x)
        x = self.conv1(x)
        # x = self.describe(x)
        x = self.conv2(x)
        # x = self.describe(x)
        x = self.mlp1(x.view(batch_size, -1))
        # x = self.describe(x)
        x = self.mlp2(x)
        # x = self.describe(x)
        return x


class ToyCNNClassifier(BaseTorchClassifier):
    _name: Final[str] = "CNN (Toy)"
    _model_type: Final[Type[AbstractTorchModule]] = ToyCNNModule


_classifiers = {
    cls._name: cls
    for cls in (
        SklearnVotingClassifier,
        SklearnSupportingVectorMachineClassifier,
        SklearnExtraTreesClassifier,
        SklearnRandomForestClassifier,
        SklearnKNearestNeighborsClassifier,
        XGBoostClassifier,
        ToyCNNClassifier
    )
}


def get_classifier_type(name: str) -> Type[AbstractClassifier]:
    return _classifiers[name]


def load_classifier(parameter_path: str, load_model: bool = True) -> AbstractClassifier:
    loaded_data = read_toml_with_metadata(parameter_path)
    return get_classifier_type(loaded_data.pop("name")).load(parameter_path, load_model=load_model)
