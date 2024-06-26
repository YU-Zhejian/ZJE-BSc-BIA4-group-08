"""
classifier -- Abstract classification models

Here contains wrapped classification algorithm from :py:mod:`sklearn`. :py:mod:`xgboost` and :py:mod:`torchvision`.
Their APIs are unified which makes frontend easier to call on them.
"""

from __future__ import annotations

import os.path
from abc import abstractmethod
from statistics import mean
from typing import TypeVar, Type, Final, Optional, Iterable, Dict, Any, Union, Generic

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
from BIA_G8.helper import ndarray_helper
from BIA_G8.helper.io_helper import read_tensor_xz, write_tensor_xz, SerializableInterface, read_toml_with_metadata, \
    write_toml_with_metadata
from BIA_G8.helper.ml_helper import MachinelearningDatasetInterface
from BIA_G8.helper.ndarray_helper import scale_np_array
from BIA_G8.model import LackingOptionalRequirementError
from BIA_G8.torch_modules import AbstractTorchModule
from BIA_G8.torch_modules.torchvision_resnet50 import TorchVisionResnet50Module
from BIA_G8.torch_modules.toy_cnn import ToyCNNModule

_lh = get_lh(__name__)

try:
    from xgboost.sklearn import XGBClassifier
except ImportError:
    XGBClassifier = None


def _classifier_interface_documentation_decorator(cls: Type[ClassifierInterface]) -> Type[ClassifierInterface]:
    cls.__doc__ = f"{cls.__name__} ({cls.name}) -- {cls.description}"
    return cls


class ClassifierInterface(SerializableInterface):
    """
    Abstract Model for extension

    The model supports pipelining like ``absc().fit(train_data).evaluate(test_data)``
    """
    name: str
    """
    Class attribute that represents human readable classifier name.
    
    :meta private:
    """

    description: str
    """
    Class attribute that represents human readable classifier description.
    
    :meta private:
    """

    @abstractmethod
    def fit(self, dataset: MachinelearningDatasetInterface) -> ClassifierInterface:
        """
        Train the module using :py:class:`MachinelearningDatasetInterface`.

        :param dataset: The training set.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, image: npt.NDArray) -> int:
        """
        Predict over one image.

        :param image: Input image, which should be an image of npt.NDArray[float64] datatype.
            at range ``[0, 1]``.
        :return: Predicted catagory.
        """
        raise NotImplementedError

    @abstractmethod
    def predicts(self, images: Iterable[npt.NDArray]) -> npt.NDArray:
        """
        Predict over a batch of images. See :py:func:`ClassifierInterface.predict()`
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, test_dataset: MachinelearningDatasetInterface) -> float:
        """
        Evaluate the model using a test dataset.

        :param test_dataset: The testing set.
        :returns: The accuracy.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def new(cls, **params):
        """
        Initialize the

        :param params: Possible parameters.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls, path: str, load_model: bool = True):
        """
        Load the model from TOML

        :param path: Source TOML path.
        :param load_model: Whether to load pretrained model (if exist).
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str, save_model: bool = True) -> None:
        """
        Save the model to TOML.

        :param path: Destination TOML path.
        :param save_model: Whether to save the trained model.
        """
        raise NotImplementedError


class DiagnosableClassifierInterface(ClassifierInterface):
    """
    Classifier that can output some diagnostic information during training.
    """

    @abstractmethod
    def diagnostic_fit(
            self,
            train_dataset: MachinelearningDatasetInterface,
            test_dataset: MachinelearningDatasetInterface,
            output_diagnoistics_path: str,
            **kwargs
    ) -> ClassifierInterface:
        """
        Fit a model with diagnostic information and validation.
        This method is used to find the best epoch number.

        :param train_dataset: Datset for training.
        :param test_dataset: Dataset for testing.
        :param output_diagnoistics_path: Path for output diagnostic file.
        """
        raise NotImplementedError


_SKLearnModelType = TypeVar("_SKLearnModelType")


class BaseSklearnClassifier(ClassifierInterface, Generic[_SKLearnModelType]):
    """
    Basic wrapper for SKLearn-compatible classifiers.
    """
    _model: _SKLearnModelType
    _model_type: Optional[Type[_SKLearnModelType]]
    _params: Dict[str, Any]

    def __init__(self, *, model: _SKLearnModelType, **params):
        """
        This method should **NOT** be called by user;
        User should use :py:func:`ClassifierInterface.new()` or :py:func:`ClassifierInterface.load()` instead.

        :param model: Model to be loaded.
            Can be a new model (as :py:func:`ClassifierInterface.new()` does)
            or model loaded from disk (as :py:func:`ClassifierInterface.load()` does).
        :param params: Possible parameters.
        """
        self._params = params
        self._model = model

    @classmethod
    def new(cls, **params):
        return cls(model=cls._model_type(**params), **params)

    def fit(self, dataset: MachinelearningDatasetInterface) -> BaseSklearnClassifier:
        self._model.fit(*dataset.sklearn_dataset)
        return self

    def predict(self, image: npt.NDArray) -> int:
        return self.predicts([image])[0]

    def predicts(self, images: Iterable[npt.NDArray]) -> npt.NDArray:
        images = np.array(list(map(
            lambda image: scale_np_array(image).ravel(),
            images
        )))
        return self._model.predict(images)

    def evaluate(self, test_dataset: MachinelearningDatasetInterface) -> float:
        x_test, y_test = test_dataset.sklearn_dataset
        y_pred = self._model.predict(x_test)
        return np.sum(y_pred == y_test) / len(y_test)

    @classmethod
    def load(cls, path: str, load_model: bool = True):
        loaded_data = read_toml_with_metadata(path)
        model_path = path + ".pkl.xz"
        if os.path.exists(model_path) and load_model:
            _lh.info("%s: Loading pretrained model...", cls.__name__)
            return cls(model=joblib.load(model_path), **loaded_data["params"])
        else:
            _lh.info("%s: Loading parameters only...", cls.__name__)
            return cls.new(**loaded_data["params"])

    def save(self, path: str, save_model: bool = True) -> None:
        _lh.info("%s: Saving...", self.__class__.__name__)
        path = os.path.abspath(path)
        out_dict = {
            "name": self.name,
            "params": self._params
        }
        if save_model:
            model_path = path + ".pkl.xz"
            joblib.dump(self._model, model_path)
        write_toml_with_metadata(out_dict, path)


@_classifier_interface_documentation_decorator
class SklearnKNearestNeighborsClassifier(BaseSklearnClassifier):
    name: Final[str] = "KNN (sklearn)"
    description: Final[str] = """
    Classifier using K Nearest Neighbors (KNN) implemented by :py:mod:`sklearn`.
    
    See :py:class:`sklearn.neighbors.KNeighborsClassifier` for more details.
    """
    _model_type: Final[_SKLearnModelType] = KNeighborsClassifier


@_classifier_interface_documentation_decorator
class SklearnSupportingVectorMachineClassifier(BaseSklearnClassifier):
    name: Final[str] = "SVM (sklearn)"
    description: Final[str] = """
    Classifier using Supporting Vector Machine/Supporting Vector Classifier (SCM/SVC) implemented by :py:mod:`sklearn`.

    See :py:class:`sklearn.svm.SVC` for more details.
    """
    _model_type: Final[_SKLearnModelType] = SVC


@_classifier_interface_documentation_decorator
class SklearnRandomForestClassifier(BaseSklearnClassifier):
    name: Final[str] = "Random Forest (sklearn)"
    description: Final[str] = """
    Classifier using Random Forest implemented by :py:mod:`sklearn`.

    See :py:class:`sklearn.ensemble.RandomForestClassifier` for more details.
    """
    _model_type: Final[_SKLearnModelType] = RandomForestClassifier


@_classifier_interface_documentation_decorator
class SklearnExtraTreesClassifier(BaseSklearnClassifier):
    name: Final[str] = "Extra Trees (sklearn)"
    description: Final[str] = """
    Classifier using Extra Trees implemented by :py:mod:`sklearn`.

    See :py:class:`sklearn.ensemble.ExtraTreesClassifier` for more details.
    """
    _model_type: Final[_SKLearnModelType] = ExtraTreesClassifier


@_classifier_interface_documentation_decorator
class SklearnVotingClassifier(BaseSklearnClassifier):
    name: Final[str] = "Voting (KNN, SVM, Decision Tree) (sklearn)"
    description: Final[str] = """
    Classifier using Voting classifier implemented by :py:mod:`sklearn`.
    
    The model is ensembled from :py:class:`sklearn.neighbors.KNeighborsClassifier`,
     :py:class:`sklearn.svm.SVC` and :py:class:`sklearn.tree.DecisionTreeClassifier`.

    See :py:class:`sklearn.ensemble.VotingClassifier` for more details.
    """
    _model_type: Final[_SKLearnModelType] = VotingClassifier

    @classmethod
    def new(cls, **params):
        new_instance = cls(model=cls._model_type(estimators=[
            ('knn', KNeighborsClassifier()),
            ('svm', SVC()),
            ('dt', DecisionTreeClassifier())
        ]))
        return new_instance


@_classifier_interface_documentation_decorator
class XGBoostClassifier(BaseSklearnClassifier):
    name: Final[str] = "Gradient Boosting Tree (XGBoost)"
    description: Final[str] = """
    Gradient Boosting Machine/Gradient Boosting Tree (GBM/GBT) implemented by XGBoost.
    
    See <https://xgboost.readthedocs.io> for more details.
    """
    _model_type: Final[_SKLearnModelType] = XGBClassifier

    @classmethod
    def new(cls, **params):
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


class BaseTorchClassifier(DiagnosableClassifierInterface):
    """
    Basic pyTorch classifier wrapper.
    """
    _model: AbstractTorchModule
    _device: str
    _num_epochs: int
    _batch_size: int
    _lr: float
    _model_type: Type[AbstractTorchModule]
    _model_params: Dict[str, Any]
    _hyper_params: Dict[str, Any]
    _should_be_convert_to_3_channels: bool
    """
    Class property indicating whether the underlying algorithm would convert 1-channel figures to 3-channel figures.
    
    Some modules from :py:mod:`torchvision` works with 3-channel figures.
    """

    @staticmethod
    def _convert_to_3_channels(batched_input: torch.Tensor) -> torch.Tensor:
        """
        ``[BATCH_SIZE, 1, WID, HEIGHT] -> [BATCH_SIZE, 3, WID, HEIGHT]``
        """
        images_in_3_channels = []
        for img in batched_input:
            images_in_3_channels.extend(torch.stack([img, img, img], dim=1))
        return torch.stack(images_in_3_channels, dim=0)

    def __init__(
            self,
            *,
            model: Union[AbstractTorchModule, nn.Module],
            hyper_params: Dict[str, Any],
            model_params: Dict[str, Any]
    ):
        self._hyper_params = hyper_params
        self._batch_size = hyper_params["batch_size"]
        self._num_epochs = hyper_params["num_epochs"]
        self._device = hyper_params["device"]
        self._lr = hyper_params["lr"]
        self._model_params = model_params
        self._model = model.to(self._device)

    @classmethod
    def load(cls, path: str, load_model: bool = True):
        loaded_data = read_toml_with_metadata(path)
        model_path = path + ".pt.xz"
        if os.path.exists(model_path) and load_model:
            _lh.info("%s: Loading pretrained model...", cls.__name__)
            return cls(
                model=read_tensor_xz(model_path),
                hyper_params=loaded_data["hyper_params"],
                model_params=loaded_data["model_params"]
            )
        else:
            _lh.info("%s: Loading parameters only...", cls.__name__)
            return cls.new(hyper_params=loaded_data["hyper_params"], model_params=loaded_data["model_params"])

    def save(self, path: str, save_model: bool = True) -> None:
        _lh.info("%s: Saving...", self.__class__.__name__)
        path = os.path.abspath(path)
        out_dict = {
            "name": self.name,
            "hyper_params": self._hyper_params,
            "model_params": self._model_params
        }
        if save_model:
            model_path = path + ".pt.xz"
            write_tensor_xz(self._model, model_path)
        write_toml_with_metadata(out_dict, path)

    def fit(self, dataset: MachinelearningDatasetInterface) -> BaseTorchClassifier:
        train_data_loader = tud.DataLoader(dataset.torch_dataset, batch_size=self._batch_size, shuffle=True)
        loss_func = nn.CrossEntropyLoss()
        opt = torch.optim.Adam(
            self._model.parameters(),
            lr=self._lr
        )
        for epoch in range(self._num_epochs):
            for i, (x_train, y_train) in enumerate(train_data_loader):

                if self._should_be_convert_to_3_channels:
                    x_train = self._convert_to_3_channels(x_train)
                x_train, y_train = x_train.to(self._device), y_train.to(self._device)
                y_train_pred_prob = self._model(x_train)
                train_loss = loss_func(y_train_pred_prob, y_train)
                opt.zero_grad()
                train_loss.backward()
                opt.step()

                y_train_pred = torch.argmax(y_train_pred_prob, dim=-1)
                if i % 10 == 0:
                    accu = (torch.sum(torch.eq(y_train_pred, y_train)) * 100 / y_train.shape[0]).item()
                    _lh.info(f"%s Epoch {epoch} batch {i}: accuracy {accu:.2f}", self.__class__.__name__)
        return self

    def predict(self, image: npt.NDArray) -> int:
        return self.predicts([image])[0]

    def predicts(self, images: Iterable[npt.NDArray]) -> npt.NDArray:
        images_torch = list(map(
            lambda image: torch.tensor(
                data=np.expand_dims(
                    ndarray_helper.scale_np_array(image),
                    axis=0
                ),
                dtype=torch.float
            ),
            images
        ))
        images_batch = torch.stack(images_torch, dim=0)

        if self._should_be_convert_to_3_channels:
            images_batch = self._convert_to_3_channels(images_batch)
        with torch.no_grad():
            return torch.argmax(self._model(images_batch.to(self._device)), dim=-1).cpu().detach().numpy()

    def diagnostic_fit(
            self,
            train_dataset: MachinelearningDatasetInterface,
            test_dataset: MachinelearningDatasetInterface,
            output_diagnoistics_path: str,
            num_epochs: Optional[int] = None
    ) -> BaseTorchClassifier:
        if num_epochs is None:
            num_epochs = self._num_epochs
        train_data_loader = tud.DataLoader(train_dataset.torch_dataset, batch_size=self._batch_size, shuffle=True)
        test_data_loader = tud.DataLoader(test_dataset.torch_dataset, batch_size=self._batch_size, shuffle=True)
        loss_func = nn.CrossEntropyLoss()
        opt = torch.optim.Adam(
            self._model.parameters(),
            lr=self._lr
        )
        with open(output_diagnoistics_path, "w") as writer:
            writer.write(",".join((
                "epoch",
                "train_loss",
                "train_accu",
                "test_loss",
                "test_accu"
            )) + "\n")
            writer.flush()
            for epoch in range(num_epochs):
                train_accumulated_tp = 0
                train_accumulated_loss = []

                for i, (x_train, y_train) in enumerate(train_data_loader):

                    if self._should_be_convert_to_3_channels:
                        x_train = self._convert_to_3_channels(x_train)
                    x_train, y_train = x_train.to(self._device), y_train.to(self._device)
                    y_train_pred_prob = self._model(x_train)
                    train_loss = loss_func(y_train_pred_prob, y_train)
                    opt.zero_grad()
                    train_loss.backward()
                    opt.step()

                    train_accumulated_loss.append(train_loss.data.item())
                    y_train_pred = torch.argmax(y_train_pred_prob, dim=-1)
                    train_accumulated_tp += torch.sum(torch.eq(y_train_pred, y_train)).item()

                test_accumulated_tp = 0
                test_accumulated_loss = []
                for i, (x_test, y_test) in enumerate(test_data_loader):

                    if self._should_be_convert_to_3_channels:
                        x_test = self._convert_to_3_channels(x_test)
                    x_test, y_test = x_test.to(self._device), y_test.to(self._device)
                    y_test_pred_prob = self._model(x_test)

                    test_loss = loss_func(y_test_pred_prob, y_test)
                    test_accumulated_loss.append(test_loss.data.item())
                    y_test_pred = torch.argmax(y_test_pred_prob, dim=-1)
                    test_accumulated_tp += torch.sum(torch.eq(y_test_pred, y_test)).item()

                _lh.info(f"%s Epoch %d finished", self.__class__.__name__, epoch)

                writer.write(",".join((
                    str(epoch),
                    str(mean(train_accumulated_loss)),
                    str(train_accumulated_tp / len(train_dataset)),
                    str(mean(test_accumulated_loss)),
                    str(test_accumulated_tp / len(test_dataset)),
                )) + "\n")
                writer.flush()
        return self

    @classmethod
    def new(
            cls,
            hyper_params: Dict[str, Any],
            model_params: Dict[str, Any]
    ):
        return cls(
            hyper_params=hyper_params,
            model_params=model_params,
            model=cls._model_type(**model_params)
        )

    def evaluate(self, test_dataset: MachinelearningDatasetInterface) -> float:
        with torch.no_grad():
            test_data_loader = tud.DataLoader(test_dataset.torch_dataset, batch_size=self._batch_size, shuffle=True)
            correct_count = 0
            for i, (x_test, y_test) in enumerate(test_data_loader):

                if self._should_be_convert_to_3_channels:
                    x_test = self._convert_to_3_channels(x_test)
                x_test, y_test = x_test.to(self._device), y_test.to(self._device)
                y_test_pred_prob = self._model(x_test)

                y_test_pred = torch.argmax(y_test_pred_prob, dim=-1)
                correct_count += torch.sum(torch.eq(y_test_pred, y_test)).item()
        return correct_count / len(test_dataset)


@_classifier_interface_documentation_decorator
class ToyCNNClassifier(BaseTorchClassifier):
    _should_be_convert_to_3_channels: Final[bool] = False
    name: Final[str] = "CNN (Toy)"
    description = "CNN classification using :py:class:`ToyCNNModule`"
    _model_type: Final[Type[AbstractTorchModule]] = ToyCNNModule


@_classifier_interface_documentation_decorator
class Resnet50Classifier(BaseTorchClassifier):
    _should_be_convert_to_3_channels: Final[bool] = True
    name: Final[str] = "CNN (Resnet 50)"
    description = "CNN classification using :py:class:`TorchVisionResnet50Module`"
    _model_type = TorchVisionResnet50Module


_classifiers = {
    cls.name: cls
    for cls in (
        SklearnVotingClassifier,
        SklearnSupportingVectorMachineClassifier,
        SklearnExtraTreesClassifier,
        SklearnRandomForestClassifier,
        SklearnKNearestNeighborsClassifier,
        XGBoostClassifier,
        ToyCNNClassifier,
        Resnet50Classifier
    )
}


def get_available_classifier_names() -> Iterable[str]:
    """Get names of all classifiers"""
    return iter(_classifiers.keys())


def get_classifier_type(name: str) -> Type[ClassifierInterface]:
    """Reflect-like mechanism that resolves name to classifier class."""
    return _classifiers[name]


def load_classifier(parameter_path: str, load_model: bool = True) -> ClassifierInterface:
    """
    Load classifier from a TOML file.

    :param parameter_path: Path of parameter TOML file to load.
    :param load_model: Whether to load pretrained model (if available).
    """
    loaded_data = read_toml_with_metadata(parameter_path)
    return get_classifier_type(loaded_data.pop("name")).load(parameter_path, load_model=load_model)
