import glob
import operator
import os
from functools import reduce
from typing import Dict

import click
import numpy.typing as npt
import skimage.io as skiio
import tqdm

from BIA_G8.helper import io_helper
from BIA_G8.model.classifier import load_classifier, ClassifierInterface
from BIA_G8.model.preprocesor_pipeline import PreprocessorPipeline

decode_dict = {
    0: "COVID",
    1: "Normal",
    2: "Viral Pneumonia"
}  # TODO: Have no idea wo convert this information, so hard coded.

VALID_IMAGE_EXTENSIONS = (
    "npy.xz",
    "png",
    "jpg",
    "jpeg",
    "tif",
    "tiff"
)


def load_data(data_path: str) -> Dict[str, npt.NDArray]:
    return {
        filename: io_helper.read_np_xz(filename) if filename.endswith(".npy.xz") else skiio.imread(filename)
        for filename in tqdm.tqdm(list(reduce(
            operator.add,
            map(
                lambda ext: list(
                    glob.glob(os.path.join(data_path, "**", f"*.{ext}"), recursive=True)
                ),
                VALID_IMAGE_EXTENSIONS
            )
        )), desc="Reading data...")
    }


def predict(
        loaded_data: Dict[str, npt.NDArray],
        loaded_pp: PreprocessorPipeline,
        loaded_classifier: ClassifierInterface
) -> Dict[str, str]:
    preprocessed_data = {
        k: loaded_pp.execute(v.copy())
        for k, v in loaded_data.items()
    }
    return {
        k: v
        for k, v in zip(
            preprocessed_data.keys(),
            map(
                lambda _k: decode_dict.get(_k, "UNKNOWN"),
                loaded_classifier.predicts(
                    preprocessed_data.values()
                ))
        )
    }


def save(predictions: Dict[str, str], output_csv: str) -> None:
    with open(output_csv, "w") as writer:
        writer.write(",".join((
            "IMAGE_PATH", "PREDICTION"
        )) + "\n")
        for filename, prediction in predictions.items():
            writer.write(",".join((
                f"\"{filename}\"", f"\"{prediction}\""
            )) + "\n")


@click.command
@click.option("--preprocessor_pipeline_config_path", help="Path to preprocessor config")
@click.option("--classifier_configuration_path", help="Path to classifier config")
@click.option("--input_image_path", help="Path to input image directory")
@click.option("--out_csv", help="Path to output CSV")
def perform_ml(
        preprocessor_pipeline_config_path: str,
        classifier_configuration_path: str,
        input_image_path: str,
        out_csv: str
) -> None:
    loaded_data = load_data(input_image_path)
    pp = PreprocessorPipeline.load(preprocessor_pipeline_config_path)
    classifier = load_classifier(classifier_configuration_path)

    predictions = predict(
        loaded_data,
        pp,
        classifier
    )
    save(predictions, out_csv)


if __name__ == "__main__":
    perform_ml()
