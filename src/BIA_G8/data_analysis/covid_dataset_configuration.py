"""
Configuration for COVID-dataset-compatible dataset.
"""

from __future__ import annotations

from typing import Dict, Any, Optional

from BIA_G8 import get_lh
from BIA_G8.data_analysis.covid_dataset import CovidDataSet
from BIA_G8.helper import ml_helper
from BIA_G8.helper.io_helper import AbstractTOMLSerializable

_lh = get_lh(__name__)


class CovidDatasetConfiguration(AbstractTOMLSerializable):
    """
    Dataset configuration.
    """
    _dataset: Optional[CovidDataSet]
    _dataset_path: str
    _encoder_dict: Dict[str, int]
    _size: int

    @property
    def dataset(self) -> CovidDataSet:
        if self._dataset is None:
            encode, decode = ml_helper.generate_encoder_decoder(self._encoder_dict)
            self._dataset = CovidDataSet.parallel_from_directory(
                dataset_path=self._dataset_path,
                encode=encode,
                decode=decode,
                n_classes=len(self._encoder_dict),
                size=self._size
            )
        return self._dataset

    def __init__(
            self,
            dataset_path: str,
            encoder_dict: Dict[str, int],
            size: int
    ):
        """
        :param dataset_path: Path to the dataset.
        :param encoder_dict: Encoder in dictionary format.
        :param size: Number of data to be loaded.
        """
        self._size = size
        self._dataset_path = dataset_path
        self._encoder_dict = dict(encoder_dict)
        self._dataset = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_path": self._dataset_path,
            "encoder_dict": self._encoder_dict,
            "size": self._size
        }

    @classmethod
    def from_dict(cls, in_dict: Dict[str, Any]):
        return cls(**in_dict)
