import numpy as np
import numpy.typing as npt

__all__ = (
    "image_mask_to_rgba",
    "is_img_rgb"
)

import BIA_KiTS19.helper.ndarray_helper


def image_mask_to_rgba(
        image: npt.NDArray,
        mask: npt.NDArray
) -> npt.NDArray:
    """
    Merge existing 3D CT image and mask into an RGBA image.

    :param image: A 3-dimension ``npt.NDArray``, representing the original image.
    :param mask: A 3-dimension ``npt.NDArray``, representing the mask. The mask should contain no more than 3 distinct values.
    :return: A 4-dimension ``npt.NDArray``, representing ``[Z, X, Y, COLOR_CHANNEL]``, whose value would be 8-bit color level.
    """
    if image.shape != mask.shape:
        raise ValueError(f"Shape mismatch image_slice: {image.shape} mask_slice: {image.shape}")
    colors = np.unique(mask.ravel())
    if len(colors) > 3:
        raise ValueError(f"At most 3 colors supported, but given {len(colors)}")
    image_rgba = np.zeros(shape=(*image.shape, 4), dtype="uint8")
    image_rgba[:, :, :, 0] = np.array(255 * (mask == colors[0]), dtype=int)
    image_rgba[:, :, :, 1] = np.array(255 * (mask == colors[1]), dtype=int)
    image_rgba[:, :, :, 2] = np.array(255 * (mask == colors[2]), dtype=int)
    image_rgba[:, :, :, 3] = np.array(
        255 * BIA_KiTS19.helper.ndarray_helper.scale_np_array(image),
        dtype=int
    )
    return image_rgba


def is_img_rgb(img: npt.NDArray) -> bool:
    """Determine whether the image is RGB"""
    if img.shape[-1] == 3:
        return True
    else:
        return False
