import click
import skimage.io as skiio

from BIA_G8.helper import io_helper
from BIA_G8.helper.ndarray_helper import scale_np_array
from BIA_G8.model.classifier import load_classifier
from BIA_G8.model.preprocesor_pipeline import PreprocessorPipeline


decode_dict = {} # TODO: Have no idea wo convert this information, so hard coded.

@click.command
@click.option("--preprocessor_pipeline_config_path", help="Path to preprocessor config")
@click.option("--classifier_configuration_paths", help="Path to classifier config")
@click.option("--input_image_path", help="Path to input image")
def perform_ml(
        preprocessor_pipeline_config_path: str,
        classifier_configuration_paths: str,
        input_image_path: str
) -> None:
    if input_image_path.endswith("npy.xz"):
        orig_img = io_helper.read_np_xz(input_image_path)
    else:
        orig_img = skiio.imread(input_image_path)
    orig_img = scale_np_array(orig_img)
    pp = PreprocessorPipeline.load(preprocessor_pipeline_config_path)
    classifier = load_classifier(classifier_configuration_paths)
    preprocessed_image = pp.execute(orig_img)
    category = classifier.predict(preprocessed_image)
    print(decode_dict[category])


if __name__ == "__main__":
    perform_ml()
