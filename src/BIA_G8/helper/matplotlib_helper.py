"""
Helper function in visualization of exploration and preprocessing steps.
"""

__all__ = (
    "plot_histogram",
)

import matplotlib.pyplot as plt
import numpy as np
from numpy import typing as npt

EMPTY_IMG = np.zeros((1, 1), dtype="uint8")
"""An 1*1 empty image"""


def plot_histogram(
        img: npt.NDArray,
        num_bins: int = 256,
        show_img: bool = True,
        log: bool = True,
        cumulative: bool = False
) -> None:
    """
    Plots a histogram of the given image

    :param img: image to be plotted
    :param num_bins: number of bins in the histogram
    :param show_img: if True, the image is plotted alongside the histograms. Require the image to be a 2d greyscale,
        RGB or RGBA image.

    :param log: if True, the histogram is plotted in log scale
    :param cumulative: if True, show cumulative histogram
    """
    fig = plt.figure(constrained_layout=True)
    shared_opts = dict(
        bins=num_bins,
        log=log,
        cumulative=cumulative
    )
    if show_img:
        subfigs = fig.subfigures(1, 2, wspace=0.07)
        ax = subfigs[0].subplots()
        ax.imshow(img)
        sf1 = subfigs[1]
    else:
        sf1 = fig.subfigures()
    if is_img_rgb(img):
        # print(sf1.subplots(3, 1))
        paxs = sf1.subplots(3, 1)
        for n, color in enumerate("rgb"):
            paxs[n].hist(img[..., n].ravel(), facecolor=color, **shared_opts)
    else:
        pax = sf1.subplots()
        pax.hist(img.ravel(), facecolor="grey", **shared_opts)


def is_img_rgb(img: npt.NDArray) -> bool:
    """Determine whether the image is RGB"""
    return img.shape[-1] == 3
