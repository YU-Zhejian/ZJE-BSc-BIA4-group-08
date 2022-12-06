import itertools
import os.path
from typing import Iterable, Dict

from BIA_G8_DATA_ANALYSIS.analysis_config import AnalysisConfiguration


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


if __name__ == "__main__":
    grid_search(
        dataset_path="/home/yuzj/Documents/2022-23-Group-08/data_analysis/covid_image_new",
        encoder_dict={
            "COVID": 0,
            "Lung_Opacity": 1,
            "Normal": 2,
            "Viral Pneumonia": 3
        },
        preprocessor_pipeline_configuration_paths=[
            "/home/yuzj/Documents/2022-23-Group-08/src/BIA_G8_DATA_ANALYSIS/pp_plain.toml",
            "/home/yuzj/Documents/2022-23-Group-08/src/BIA_G8_DATA_ANALYSIS/pp_adapt_hist.toml",
            "/home/yuzj/Documents/2022-23-Group-08/src/BIA_G8_DATA_ANALYSIS/pp_unsharp.toml",
        ],
        classifier_configuration_paths=[
            "/home/yuzj/Documents/2022-23-Group-08/src/BIA_G8_DATA_ANALYSIS/cnn.toml",
            "/home/yuzj/Documents/2022-23-Group-08/src/BIA_G8_DATA_ANALYSIS/vote.toml",
            "/home/yuzj/Documents/2022-23-Group-08/src/BIA_G8_DATA_ANALYSIS/xgb.toml",
            "/home/yuzj/Documents/2022-23-Group-08/src/BIA_G8_DATA_ANALYSIS/knn.toml",
            "/home/yuzj/Documents/2022-23-Group-08/src/BIA_G8_DATA_ANALYSIS/svc.toml",
            "/home/yuzj/Documents/2022-23-Group-08/src/BIA_G8_DATA_ANALYSIS/extra_trees.toml",
            "/home/yuzj/Documents/2022-23-Group-08/src/BIA_G8_DATA_ANALYSIS/rf.toml",
        ],
        n_data_to_load=600,
        n_classes=4,
        out_csv="gs.csv",
        replication=10
    )
