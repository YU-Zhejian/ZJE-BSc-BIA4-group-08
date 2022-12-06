"""
Configuration for grid search.
"""

from __future__ import annotations

import itertools
import os
from typing import Dict, Any, Iterable

from BIA_G8 import get_lh
from BIA_G8.data_analysis.covid_dataset import CovidDataSet
from BIA_G8.helper import ml_helper
from BIA_G8.helper.io_helper import AbstractTOMLSerializable
from BIA_G8.model.classifier import ClassifierInterface, load_classifier
from BIA_G8.model.preprocesor_pipeline import PreprocessorPipeline

_lh = get_lh(__name__)


class AnalysisConfiguration(AbstractTOMLSerializable):
    """
    Analysis configuration that consists a preprocessor pipeline configuration and a classifier configuration.
    """
    _dataset: CovidDataSet
    _dataset_path: str
    _encoder_dict: Dict[str, int]
    _preprocessing_pipeline: PreprocessorPipeline
    _preprocessing_pipeline_configuration_path: str
    _classifier: ClassifierInterface
    _classifier_configuration_path: str
    _size: int

    def __init__(
            self,
            dataset_path: str,
            encoder_dict: Dict[str, int],
            preprocessor_pipeline_configuration_path: str,
            classifier_configuration_path: str,
            size: int,
            load_pretrained_model: bool = False
    ):
        """
        :param dataset_path: Path to the dataset.
        :param encoder_dict: Encoder in dictionary format.
        :param preprocessor_pipeline_configuration_path: Path to preprocessor pipeline configuration.
        :param classifier_configuration_path: Path to classifier configuration.
        :param size: Number of data to be loaded.
        :param load_pretrained_model: Whether to load pretrained model.
        """
        self._size = size
        self._dataset_path = dataset_path
        self._encoder_dict = dict(encoder_dict)
        encode, decode = ml_helper.generate_encoder_decoder(self._encoder_dict)
        self._dataset = CovidDataSet.parallel_from_directory(
            dataset_path=self._dataset_path,
            encode=encode,
            decode=decode,
            n_classes=len(self._encoder_dict),
            size=self._size
        )
        self._preprocessing_pipeline_configuration_path = preprocessor_pipeline_configuration_path
        self._preprocessing_pipeline = PreprocessorPipeline.load(preprocessor_pipeline_configuration_path)
        self._classifier_configuration_path = classifier_configuration_path
        self._classifier = load_classifier(self._classifier_configuration_path, load_model=load_pretrained_model)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_path": self._dataset_path,
            "encoder_dict": self._encoder_dict,
            "preprocessor_pipeline_configuration_path": self._preprocessing_pipeline_configuration_path,
            "classifier_configuration_path": self._classifier_configuration_path,
            "size": self._size
        }

    @classmethod
    def from_dict(cls, in_dict: Dict[str, Any]) -> AbstractTOMLSerializable:
        return cls(**in_dict)

    def pre_process(self) -> AnalysisConfiguration:
        """Execute preprocessor. This step can be chained."""
        _lh.info("Preprocessing...")
        self._dataset = self._dataset.parallel_apply(
            self._preprocessing_pipeline.execute,
            desc="Applying preprocessing steps..."
        )
        _lh.info("Preprocessing Done")
        return self

    def ml(self) -> float:
        """
        Execute classifier.

        :return: Accuracy
        """
        _lh.info("Splitting dataset...")
        ds_train, ds_test = self._dataset.train_test_split()
        _lh.info("Training...")
        self._classifier = self._classifier.fit(ds_train)
        _lh.info("Evaluating...")
        accuracy = self._classifier.evaluate(ds_test)
        _lh.info(
            "Evaluation finished with accuracy=%.2f%%",
            accuracy * 100
        )
        return accuracy


def grid_search(
        preprocessor_pipeline_configuration_paths: Iterable[str],
        classifier_configuration_paths: Iterable[str],
        out_csv: str,
        replication: int,
        **kwargs
) -> None:
    """
    Grid search using multiple preprocessor config and classifier config. The model will **NOT** be saved.

    :param preprocessor_pipeline_configuration_paths: Paths to preprocessor pipeline TOMLs.
    :param classifier_configuration_paths: Paths to classifier configurations.
    :param out_csv: Path to output CSV.
    :param replication: Number of replications of each method.
    :param kwargs: Arguments for :py:class:`AnalysisConfiguration`
    """
    with open(out_csv, "w") as writer:
        writer.write(",".join((
            "replication",
            "preprocessor_pipeline_configuration_path",
            "classifier_configuration_path",
            "accuracy"
        )) + "\n")
        for (
                preprocessor_pipeline_configuration_path,
                classifier_configuration_path
        ) in itertools.product(
            preprocessor_pipeline_configuration_paths,
            classifier_configuration_paths
        ):
            for i in range(replication):
                accu = AnalysisConfiguration(
                    preprocessor_pipeline_configuration_path=preprocessor_pipeline_configuration_path,
                    classifier_configuration_path=classifier_configuration_path, load_pretrained_model=False,
                    **kwargs).pre_process().ml()
                writer.write(",".join((
                    str(i),
                    os.path.basename(preprocessor_pipeline_configuration_path),
                    os.path.basename(classifier_configuration_path),
                    str(accu)
                )) + "\n")
                writer.flush()
