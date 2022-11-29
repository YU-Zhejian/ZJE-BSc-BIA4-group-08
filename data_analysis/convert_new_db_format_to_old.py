from glob import glob
import os
import skimage.io as skiio
import skimage.transform as skitrans
import skimage.exposure as skiexp
import tqdm

from BIA_G8 import get_lh
from BIA_G8.helper.io_helper import write_np_xz
from BIA_G8.helper.joblib_helper import parallel_map

lh = get_lh(__name__)

def convert(_image_path:str, _mask_path:str, _out_dir:str) -> None:
    lh.debug("Converting %s", _image_path)
    final_image_size = (256, 256)
    image, mask = skiio.imread(_image_path), skiio.imread(_mask_path)
    # Convert the images and masks in to a 2D image with same size
    if len(image.shape) == 3:
        image = image[:,:,0]
    
    if len(mask.shape) == 3:
        mask = mask[:,:,0]
    
    if image.shape !=final_image_size:
        image = skitrans.resize(image, final_image_size)
    
    if mask.shape !=final_image_size:
        mask = skitrans.resize(image, final_image_size)
    # Retain the lung region while color other parts of the images into black and enhance it with adapthist
    out_image = skiexp.equalize_adapthist(skiexp.rescale_intensity(image * mask))

    # Save the out_images into foders named by their lables under out_dir.
    paths = _image_path.split(os.path.sep)
    label_name, image_name = paths[-2], paths[-1]
    os.makedirs(os.path.join(out_dir, label_name), exist_ok=True)
    # Here we use .npy.xz as the format of saving because saving the image with skimage.io requir changing image format which can reduce the quality of the image.
    write_np_xz(out_image, os.path.join(_out_dir, label_name, image_name.replace(".png", ".npy.xz")))

if __name__ == "__main__":

    images_dir='D:\\BIA-G8\\src\\ipynb\\COVID-19_Radiography_Dataset\\images'
    out_dir = 'D:\\BIA-G8\\src\\ipynb\\COVID-19_Radiography_Dataset\\out'

    image_dirnames = list(glob(os.path.join(images_dir, "*", "")))
    mask_dirnames = [dirname.replace("images", "masks") for dirname in image_dirnames]
    # Sort the objects in the same order.
    for image_dirname, mask_dirname in zip(image_dirnames, mask_dirnames):
        image_paths = sorted(glob(os.path.join(image_dirname, "*.png")))
        mask_paths = sorted(glob(os.path.join(mask_dirname, "*.png")))
        # Increase the speed
        parallel_map(
            #Convert these images to to masked ones.
            lambda image_and_mask_path: convert(*image_and_mask_path, out_dir),
            # Add progress bar.
            tqdm.tqdm(
                iterable=list(zip(image_paths, mask_paths)),
                desc=f"Converting images from {image_dirname}"
            )
        )
        
