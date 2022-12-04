from typing import Dict, Any

from BIA_G8 import get_lh
from BIA_G8.helper import ml_helper
from BIA_G8.helper.io_helper import AbstractTOMLSerializable
from BIA_G8.model.classifier import AbstractClassifier
from BIA_G8.model.preprocesor_pipeline import PreprocessorPipeline
from BIA_G8_DATA_ANALYSIS.covid_dataset import CovidDataSet

_lh = get_lh(__name__)


class AnalysisConfiguration(AbstractTOMLSerializable):
    _dataset: CovidDataSet
    _dataset_path: str
    _encoder_dict: Dict[str, int]
    _preprocessing_pipeline: PreprocessorPipeline
    _preprocessing_pipeline_configuration_path: str
    _classifier: AbstractClassifier

    @property
    def dataset_path(self) -> str:
        return self._dataset_path

    @property
    def encoder_dict(self) -> Dict[str, int]:
        return dict(self._encoder_dict)

    @property
    def preprocessor_pipeline_configuration(self) -> PreprocessorPipeline:
        return self._preprocessing_pipeline

    def __init__(
            self,
            dataset_path: str,
            encoder_dict: Dict[str, int],
            preprocessor_pipeline_configuration_path: str
    ):
        self._dataset_path = dataset_path
        self._encoder_dict = dict(encoder_dict)
        encode, decode = ml_helper.generate_encoder_decoder(self._encoder_dict)
        self._dataset = CovidDataSet.parallel_from_directory(
            dataset_path=self._dataset_path,
            encode=encode, decode=decode
        )
        self._preprocessor_pipeline_configuration = preprocessor_pipeline_configuration_path
        self._preprocessing_pipeline = PreprocessorPipeline.load(
            preprocessor_pipeline_configuration_path
        )
        self._classifier = AbstractClassifier.new()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_path": self._dataset_path,
            "encoder_dict": self._encoder_dict,
            "preprocessor_pipeline_configuration_path": self._preprocessing_pipeline_configuration_path
        }

    @classmethod
    def from_dict(cls, in_dict: Dict[str, Any]) -> AbstractTOMLSerializable:
        return cls(**in_dict)

    def pre_process(self):
        self._dataset = self._dataset.parallel_apply(
            self._preprocessing_pipeline.execute
        )

    def ml(self) -> float:
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
