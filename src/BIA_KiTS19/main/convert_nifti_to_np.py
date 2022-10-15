import os

import click
import tqdm

from BIA_KiTS19 import get_lh
from BIA_KiTS19.helper import dataset_helper, joblib_helper

_lh = get_lh(__name__)


def _ensure_np_image(image_set: dataset_helper.ImageSet):
    _lh.info("Processing images at %s", image_set.image_dir)
    _ = image_set.np_image_final
    _ = image_set.np_mask_final
    _ = image_set.tensor_image_final
    _ = image_set.tensor_mask_final
    image_set.clear_all_cache()
    _lh.info("Processing images at %s FIN", image_set.image_dir)


@click.command()
@click.option(
    "--data_dir",
    default=os.getcwd(),
    help="The directory where all cases (i.e., folders started with case_00000 -> case_00209) are stored"
)
def main(data_dir: str):
    _lh.info("Loading dataset at %s", data_dir)
    dataset = dataset_helper.DataSet(data_dir=data_dir)
    _ = list(joblib_helper.parallel_map(_ensure_np_image, tqdm.tqdm(iterable=dataset), n_jobs=10))
    _lh.info("Loading dataset at %s FIN", data_dir)


if __name__ == "__main__":
    main()
