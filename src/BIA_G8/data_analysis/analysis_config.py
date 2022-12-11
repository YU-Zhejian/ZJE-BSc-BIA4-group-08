"""
Configuration for grid search.
"""

from __future__ import annotations

import itertools
import os
from typing import Iterable

from BIA_G8 import get_lh
from BIA_G8.data_analysis.covid_dataset import CovidDataSet
from BIA_G8.data_analysis.covid_dataset_configuration import CovidDatasetConfiguration
from BIA_G8.model.classifier import ClassifierInterface, load_classifier
from BIA_G8.model.preprocesor_pipeline import PreprocessorPipeline

_lh = get_lh(__name__)


class AnalysisConfiguration:
    """
    Analysis configuration that consists a preprocessor pipeline configuration and a classifier configuration.
    """

    _preprocessing_pipeline: PreprocessorPipeline
    _preprocessing_pipeline_configuration_path: str
    _classifier: ClassifierInterface
    _classifier_configuration_path: str
    _dataset: CovidDataSet
    _dataset_configuration_path: str

    def __init__(
            self,
            preprocessor_pipeline_configuration_path: str,
            classifier_configuration_path: str,
            dataset_configuration_path: str,
    ):
        """
        :param dataset_configuration_path: Path to dataset configuration.
        :param preprocessor_pipeline_configuration_path: Path to preprocessor pipeline configuration.
        :param classifier_configuration_path: Path to classifier configuration.
        """
        self._dataset_configuration_path = dataset_configuration_path
        self._dataset = CovidDatasetConfiguration.load(dataset_configuration_path).dataset
        self._preprocessing_pipeline_configuration_path = preprocessor_pipeline_configuration_path
        self._preprocessing_pipeline = PreprocessorPipeline.load(preprocessor_pipeline_configuration_path)
        self._classifier_configuration_path = classifier_configuration_path
        self._classifier = load_classifier(self._classifier_configuration_path, load_model=False)

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
        dataset_configuration_paths: Iterable[str],
        out_csv: str,
        replication: int,
) -> None:
    """
    Grid search using multiple preprocessor config and classifier config. The model will **NOT** be saved.

    :param preprocessor_pipeline_configuration_paths: Paths to preprocessor pipeline TOMLs.
    :param classifier_configuration_paths: Paths to classifier configurations.
    :param out_csv: Path to output CSV.
    :param replication: Number of replications of each method.
    :param dataset_configuration_paths: Paths to dataset configuration.
    """
    with open(out_csv, "w") as writer:
        writer.write(",".join((
            "replication",
            "preprocessor_pipeline_configuration_path",
            "classifier_configuration_path",
            "dataset_configuration_path",
            "accuracy"
        )) + "\n")
        for (
                preprocessor_pipeline_configuration_path,
                classifier_configuration_path,
                dataset_configuration_path
        ) in itertools.product(
            preprocessor_pipeline_configuration_paths,
            classifier_configuration_paths,
            dataset_configuration_paths
        ):
            for i in range(replication):
                accu = AnalysisConfiguration(
                    preprocessor_pipeline_configuration_path=preprocessor_pipeline_configuration_path,
                    classifier_configuration_path=classifier_configuration_path,
                    dataset_configuration_path=dataset_configuration_path
                ).pre_process().ml()
                writer.write(",".join((
                    str(i),
                    os.path.basename(preprocessor_pipeline_configuration_path),
                    os.path.basename(classifier_configuration_path),
                    os.path.basename(dataset_configuration_path),
                    str(accu)
                )) + "\n")
                writer.flush()
