import glob
import os

import numpy.typing as npt
import skimage.color as skicol
import skimage.io as skiio
import skimage.transform as skitrans
import tqdm

from BIA_G8.helper import io_helper, matplotlib_helper, joblib_helper


def convert(img: npt.NDArray) -> npt.NDArray:
    desired_size = (1024, 1024)
    if matplotlib_helper.is_img_rgb(img):
        img = skicol.rgb2gray(img)
    if img.shape != desired_size:
        img = skitrans.resize(img, desired_size)
    return img


def convert_path(
        source_image_path: str,
        dest_path: str,
        label: str,
        i: int
):
    converte_img = convert(skiio.imread(source_image_path))
    io_helper.write_np_xz(
        converte_img,
        os.path.join(dest_path, label, f"{i}.npy.xz")
    )
    # skiio.imsave(os.path.join(dest_path, f"{label}_{_uuid}.tiff"), converte_img)


def main(source_path: str, dest_path: str):
    os.makedirs(dest_path, exist_ok=True)
    for directory_path in glob.glob(os.path.join(source_path, "*", "")):
        label = os.path.basename(directory_path.rstrip(os.sep)).replace(" ", "_")
        os.makedirs(os.path.join(dest_path, label), exist_ok=True)
        _ = list(joblib_helper.parallel_map(
            lambda x: convert_path(x[1], dest_path, label, x[0]),
            enumerate(
                tqdm.tqdm(
                    iterable=glob.glob(os.path.join(directory_path, "*.png")),
                    desc=f"Converting images from label {label}"
                )
            )
        ))


if __name__ == '__main__':
    main("/media/yuzj/BUP/covid19-database", "/media/yuzj/BUP/covid19-database-np")
