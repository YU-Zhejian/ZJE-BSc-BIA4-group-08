"""
General-purposed machine learning helpers.
"""

from __future__ import annotations

__all__ = (
    "print_confusion_matrix",
    "generate_encoder_decoder",
    "MachinelearningDatasetInterface"
)

from abc import abstractmethod
from typing import Dict, Tuple, Callable, Iterable

import numpy as np
import numpy.typing as npt
import prettytable

from BIA_G8.helper import torch_helper


def generate_encoder_decoder(
        encoder_dict: Dict[str, int]
) -> Tuple[Callable[[str], int], Callable[[int], str]]:
    """Generate encoding and decoding function"""
    decoder_dict = {v: k for k, v in encoder_dict.items()}

    def encoder(label_str: str) -> int:
        return encoder_dict[label_str]

    def decoder(label: int) -> str:
        return decoder_dict[label]

    return encoder, decoder


def print_confusion_matrix(
        matrix: npt.NDArray,
        labels: Iterable[str]
) -> str:
    """
    Print confusion matrix using ``prettytable`` package.

    Example:

    >>> print(print_confusion_matrix(np.array([[1, 0, 0], [0, 5, 2], [0, 0, 9]]), labels=["a", "b", "c"]))
    +-------+---+---+---+
    | title | a | b | c |
    +-------+---+---+---+
    |   a   | 1 | 0 | 0 |
    |   b   | 0 | 5 | 2 |
    |   c   | 0 | 0 | 9 |
    +-------+---+---+---+

    :param matrix: The confusion matrix.
    :param labels: Row and Column names.
    :return: Confusion matrix in string.
    """
    labels = list(labels)
    table_length = matrix.shape[0]
    field_names = list(map(labels.__getitem__, range(table_length)))
    pt = prettytable.PrettyTable(("title", *field_names))
    for i in range(table_length):
        pt.add_row((labels[i], *matrix[i]))
    return str(pt)


class MachinelearningDatasetInterface:
    """
    Dataset that supports applying machine learning algorithms.
    """

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def sklearn_dataset(self) -> Tuple[npt.NDArray, npt.NDArray]:
        """
        Prepare and return cached dataset for :py:mod:`sklearn`.

        :return: A tuple of ``X`` and ``y`` for :py:func:`fit`-like functions.
            For example, as is used in :external+sklearn:py:class:`sklearn.neighbors.KNeighborsClassifier`.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def torch_dataset(self) -> torch_helper.DictBackedTorchDataSet:
        """
        Prepare and return cached dataset for :py:mod:`torch`.

        :return: An iterable pytorch dataset.
        """
        raise NotImplementedError

    @abstractmethod
    def train_test_split(self, ratio: float = 0.7) -> Tuple[
        MachinelearningDatasetInterface,
        MachinelearningDatasetInterface
    ]:
        """
        Split current dataset into training and testing dataset.

        :param ratio: Train-test ratio.
        :return: Two new datasets
        """
        raise NotImplementedError
