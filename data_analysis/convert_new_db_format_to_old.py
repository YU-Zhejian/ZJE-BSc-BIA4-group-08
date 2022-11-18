from glob import glob
import os
import skimage.io as skiio
import skimage.transform as skitrans
import tqdm
import click

from BIA_G8 import get_lh
from BIA_G8.helper.io_helper import write_np_xz
from BIA_G8.helper.joblib_helper import parallel_map

lh = get_lh(__name__)


def convert(_image_path: str, _mask_path: str, _out_dir: str) -> None:
    lh.debug("Converting %s", _image_path)
    final_image_size = (256, 256)
    image, mask = skiio.imread(_image_path), skiio.imread(_mask_path)

    if len(image.shape) == 3:
        image = image[:, :, 0]

    if len(mask.shape) == 3:
        mask = mask[:, :, 0]

    if image.shape != final_image_size:
        image = skitrans.resize(image, final_image_size)

    if mask.shape != final_image_size:
        mask = skitrans.resize(image, final_image_size)

    out_image = image * mask
    paths = _image_path.split(os.path.sep)
    label_name, image_name = paths[-2], paths[-1]
    os.makedirs(os.path.join(_out_dir, label_name), exist_ok=True)
    write_np_xz(out_image, os.path.join(_out_dir, label_name, image_name.replace(".png", ".npy.xz")))


@click.command()
@click.option("--images_dir", help="Database using new format")
@click.option("--out_dir", help="Database using old format")
def main(images_dir: str, out_dir: str) -> None:
    image_dirnames = list(glob(os.path.join(images_dir, "*", "")))
    mask_dirnames = [dirname.replace("images", "masks") for dirname in image_dirnames]
    for image_dirname, mask_dirname in zip(image_dirnames, mask_dirnames):
        image_paths = sorted(glob(os.path.join(image_dirname, "*.png")))
        mask_paths = sorted(glob(os.path.join(mask_dirname, "*.png")))
        parallel_map(
            lambda image_and_mask_path: convert(*image_and_mask_path, out_dir),
            tqdm.tqdm(
                iterable=list(zip(image_paths, mask_paths)),
                desc=f"Converting images from {image_dirname}"
            )
        )


if __name__ == "__main__":
    main()
