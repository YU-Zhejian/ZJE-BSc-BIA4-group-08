import itertools

import k3d
import numpy as np
import numpy.typing as npt


def rgb_to_hex(r: int, g: int, b: int) -> int:
    return r * 256 * 256 + g * 256 + b


def image_to_k3d_points(
        image: npt.NDArray
) -> k3d.objects.Points:
    positions = list(itertools.product(
        *map(np.arange, image.shape)
    ))
    opacities = image.ravel() / 256
    return k3d.points(
        positions=positions,
        color=0xffffff,
        opacities=opacities
    )


def image_mask_rgba_to_k3d_points(
        image_mask_rgba: npt.NDArray
) -> k3d.objects.Points:
    positions = list(itertools.product(
        map(range, image_mask_rgba.shape)
    ))
    colors = list(map(
        lambda rgb: rgb_to_hex(*rgb),
        map(
            lambda position: image_mask_rgba[position[0], position[1], position[2], 0:2],
            positions
        )
    ))
    opacities = list(map(
        lambda position: image_mask_rgba[position[0], position[1], position[2], 3],
        positions
    ))
    return k3d.points(
        positions=positions,
        colors=colors,
        opacities=opacities
    )
