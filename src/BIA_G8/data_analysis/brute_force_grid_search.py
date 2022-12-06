import itertools
import os.path
from typing import Iterable, Dict

from BIA_G8.data_analysis.analysis_config import AnalysisConfiguration


def grid_search(
        preprocessor_pipeline_configuration_paths: Iterable[str],
        classifier_configuration_paths: Iterable[str],
        dataset_path: str,
        encoder_dict: Dict[str, int],
        n_data_to_load: int,
        n_classes: int,
        out_csv: str,
        replication: int
):
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
                    dataset_path=dataset_path,
                    encoder_dict=encoder_dict,
                    n_data_to_load=n_data_to_load,
                    n_classes=n_classes,
                    preprocessor_pipeline_configuration_path=preprocessor_pipeline_configuration_path,
                    classifier_configuration_path=classifier_configuration_path,
                    load_pretrained_model=False
                ).pre_process().ml()
                writer.write(",".join((
                    str(i),
                    os.path.basename(preprocessor_pipeline_configuration_path),
                    os.path.basename(classifier_configuration_path),
                    str(accu)
                )) + "\n")
                writer.flush()
