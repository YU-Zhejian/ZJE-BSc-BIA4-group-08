from glob import glob
import os
import skimage.io as skiio
import skimage.transform as skitrans
import tqdm

from BIA_G8 import get_lh
from BIA_G8.helper.io_helper import write_np_xz
from BIA_G8.helper.joblib_helper import parallel_map

lh = get_lh(__name__)

def convert(_image_path:str, _mask_path:str, _out_dir:str) -> None:
    lh.debug("Converting %s", _image_path)
    image, mask = skiio.imread(_image_path), skitrans.resize(skiio.imread(_mask_path)[:,:,0],(299,299))
    out_image = image * mask
    paths = _image_path.split(os.path.sep)
    _, label_name, image_name = os.path.join(*paths[0:-3]), paths[-2], paths[-1]
    os.makedirs(os.path.join(out_dir, label_name), exist_ok=True)
    write_np_xz(out_image, os.path.join(_out_dir, label_name, image_name.replace(".png", ".npy.xz")))

if __name__ == "__main__":

    images_dir='D:\\BIA-G8\\src\\ipynb\\COVID-19_Radiography_Dataset\\images'
    out_dir = 'D:\\BIA-G8\\src\\ipynb\\COVID-19_Radiography_Dataset\\out'

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
        # for image_path, mask_path in zip(image_paths, mask_paths):
        #     convert(image_path, mask_path, out_dir)