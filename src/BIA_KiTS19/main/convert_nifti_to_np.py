import os

import click
import tqdm

from BIA_KiTS19.helper import dataset_helper


def _ensure_np_image(image_set: dataset_helper.ImageSet):
    _ = image_set.space_resampled_np_image
    _ = image_set.space_resampled_np_mask
    image_set.clear_all_cache()


@click.command()
@click.option(
    "--data_dir",
    default=os.getcwd(),
    help="The directory where all cases (i.e., folders started with case_000000 -> case_000209) are stored"
)
def main(data_dir: str):
    dataset = dataset_helper.DataSet(data_dir=data_dir)
    _ = list(map(_ensure_np_image, tqdm.tqdm(iterable=dataset)))


if __name__ == "__main__":
    main()
