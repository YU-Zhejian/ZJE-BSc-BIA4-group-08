from typing import Iterable

import numpy as np
import numpy.typing as npt


def image_mask_slice_to_rgba(
        image_slice: npt.NDArray,
        mask_slice: npt.NDArray
) -> npt.NDArray:
    """
    pass

    :param image_slice: A 2-dimension ``npt.NDArray``, representing a slice of the original image.
    The imaghe should be in ``uint8`` data type.
    :param mask_slice: A 2-dimension ``npt.NDArray``, representing a slice of the mask.
    The mask should contain no more than 3 distinct values.
    :return: A 3-dimension ``npt.NDArray``, representing ``[X, Y, COLOR_CHANNEL]``,
    whose value would be color level.
    """
    if image_slice.shape != mask_slice.shape:
        raise ValueError(f"Shape mismatch image_slice: {image_slice.shape} mask_slice: {mask_slice.shape}")
    colors = np.unique(mask_slice.ravel())
    if len(colors) > 3:
        raise ValueError(f"At most 3 colors supported, but given {len(colors)}")
    image_slice_with_rgba = np.zeros(shape=(*image_slice.shape, 4), dtype="uint8")
    image_slice_with_rgba[:, :, 0] = 255 * (mask_slice == colors[0])
    image_slice_with_rgba[:, :, 1] = 255 * (mask_slice == colors[1])
    image_slice_with_rgba[:, :, 2] = 255 * (mask_slice == colors[2])
    image_slice_with_rgba[:, :, 3] = image_slice
    return image_slice_with_rgba


def get_slice(
        img: npt.NDArray,
        axis: int,
        n: int
):
    if axis == 0:
        return img[n, :, :]
    elif axis == 1:
        return img[:, n, :]
    elif axis == 2:
        return img[:, :, n]
    else:
        raise ValueError(f"Illegal axis {axis} which should be in [0, 1, 2]")


def sample_along(
        img: npt.NDArray,
        axis: int = 0,
        start: int = 0,
        end: int = -1,
        step: int = 0
) -> Iterable[npt.NDArray]:
    if end == -1:
        end = img.size[axis]
    while start < end:
        yield get_slice(img=img, axis=axis, n=start)
        start += step


def image_mask_to_rgba(
        image: npt.NDArray,
        mask: npt.NDArray
) -> npt.NDArray:
    """
    pass

    :param image_slice: A 2-dimension ``npt.NDArray``, representing the original image.
    :param mask_slice: A 2-dimension ``npt.NDArray``, representing the mask.
    The mask should contain no more than 3 distinct values.
    :return: A 4-dimension ``npt.NDArray``, representing ``[Z, X, Y, COLOR_CHANNEL]``,
    whose value would be color level.
    """
    image_uint8 = np.array(image, dtype="uint8")
    image_with_rgba = np.zeros(shape=(image_uint8.shape, 4), dtype="uint8")
    for index, image_slice, mask_slice in enumerate(zip(
            sample_along(image_uint8), sample_along(mask)
    )):
        image_with_rgba[index, :, :, :, :] = image_mask_slice_to_rgba(image_slice, mask_slice)
    return image_with_rgba
