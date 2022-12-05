"""
Here provides compressed readers and writers for Pickle, Numpy and Torch serialization formats,
which can significantly reduce disk size.

We also have an abstract base class that allows programmers to create their own configuration class.

The compression algorithm would be Lempel-Ziv Markov Chain Algorithm (LZMA) version 2 used in
`7-Zip <https://www.7-zip.org>`_. The implementation is provided Python standard library :py:mod:`lzma`.

.. warning::
    Since Python's standard LZMA implementation is single-threaded, it might be extremely slow to compress large objects!
"""

from __future__ import annotations

__all__ = (
    "read_np_xz",
    "read_tensor_xz",
    "write_np_xz",
    "write_tensor_xz",
    "SerializableInterface",
    "AbstractTOMLSerializable"
)

import lzma
import pickle
from abc import abstractmethod, ABC
from typing import Any, Union, Mapping, Dict

import numpy as np
import numpy.lib.format as npy_format
import numpy.typing as npt
import tomli
import tomli_w
import torch
from torch import nn

from BIA_G8.helper.metadata_helper import validate_versions, dump_versions, dump_metadata


def read_np_xz(path: str) -> npt.NDArray[Any]:
    """Reader of compressed Numpy serialization format"""
    with lzma.open(path, "rb") as reader:
        return npy_format.read_array(reader)


def read_tensor_xz(path: str) -> Union[torch.Tensor, Mapping[str, Any], nn.Module]:
    """Reader of compressed Torch serialization format"""
    with lzma.open(path, "rb") as reader:
        return torch.load(reader)


def read_pickle_xz(path: str) -> Any:
    with lzma.open(path, "rb") as reader:
        return pickle.load(reader)


def write_np_xz(array: npt.NDArray[Any], path: str) -> None:
    """Writer of compressed Numpy serialization format"""
    with lzma.open(path, "wb", preset=9) as writer:
        npy_format.write_array(writer, np.asanyarray(array))


def write_tensor_xz(array: Union[torch.Tensor, Mapping[str, Any], nn.Module], path: str) -> None:
    """Writer of compressed Torch serialization format"""
    with lzma.open(path, "wb", preset=9) as writer:
        torch.save(array, writer)


def write_pickle_xz(obj: Any, path: str) -> None:
    with lzma.open(path, "wb", preset=9) as writer:
        pickle.dump(obj, writer)


class SerializableInterface:
    @classmethod
    def load(cls, path: str) -> SerializableInterface:
        """
        Load configuration from a file.

        :param path: Filename to read from.
        :return: New instance of corresponding class.
        """
        pass

    def save(self, path: str) -> None:
        """
        Save the class contents with metadata.

        :param path: Filename to write to.
        """
        pass


class TOMLSerializableInterface:
    """
    Abstract Base Class of something that can be represented as TOML.

    Should be used as configuration class.
    """

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Dump the item to a dictionary"""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_dict(cls, in_dict: Dict[str, Any]) -> TOMLSerializableInterface:
        """Load the item from a dictionary"""
        raise NotImplementedError


def read_toml_with_metadata(path: str) -> Dict[str, Any]:
    with open(path, "rb") as reader:
        in_dict = tomli.load(reader)
    if "version_info" in in_dict:
        validate_versions(in_dict.pop("version_info"))
    if "metadata" in in_dict:
        _ = in_dict.pop("metadata")
    return in_dict


def write_toml_with_metadata(obj: Dict[str, Any], path: str) -> None:
    retd = dict(obj)
    retd["version_info"] = dump_versions()
    retd["metadata"] = dump_metadata()
    with open(path, 'wb') as writer:
        tomli_w.dump(retd, writer)


class AbstractTOMLSerializable(
    SerializableInterface,
    TOMLSerializableInterface,
    ABC
):
    """
    Abstract Base Class of something that can be represented as TOML.

    Should be used as configuration class.
    """

    @classmethod
    def load(cls, path: str) -> AbstractTOMLSerializable:
        return cls.from_dict(read_toml_with_metadata(path))

    def save(self, path: str) -> None:
        write_toml_with_metadata(self.to_dict(), path)
