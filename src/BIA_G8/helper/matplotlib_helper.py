"""
Helper function in visualization of explorization and preprocessing steps.
"""

__all__ = (
    "plot_3d_rgba",
    "plot_histogram"
)

from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from numpy import typing as npt

from BIA_G8.helper import ndarray_helper

EMPTY_IMG = np.zeros((1, 1), dtype="uint8")
"""An 1*1 empty image"""


def plot_3d_rgba(
        img_3d_rgba: npt.NDArray,
        num_slices: int = 9,
        ncols: int = 3,
        axis: int = 0,
        width: Union[int, float] = 10
):
    """
    Slice and plot 3d greyscale, RGB or RGBA images.

    Coloring: If the image is greyscale, the color map would be `"bone"`. Otherwise, would show an RGB image on black
    background.

    :param img_3d_rgba: 3d greyscale, RGB or RGBA images
    :param num_slices: Number of slices to take. Recommended to be a
    :param ncols: Number oif columns. Recommended to be less than 5.
    :param axis: Axis to be sliced. Can be `0`, `1` or `2`.
    :param width: Width of produced image in inches.
    """
    z_len = img_3d_rgba.shape[axis]
    step = z_len // num_slices
    downsampled_img = ndarray_helper.sample_along_np(array=img_3d_rgba, axis=axis, start=0, end=z_len, step=step)
    axs: npt.NDArray[plt.Axes]
    fig: plt.Figure
    fig, axs = plt.subplots(nrows=num_slices // ncols, ncols=ncols)
    fig.set_facecolor("black")
    if is_img_rgb(img_3d_rgba):
        cmap = None
    else:
        cmap = "bone"
    for index, ax in enumerate(axs.ravel()):
        ax: plt.Axes
        ax.minorticks_off()
        ax.grid(visible=False)
        ax.set_facecolor("black")
        ax.set_frame_on(True)
        plt.setp(ax.spines.values(), color="white")
        plt.setp([ax.get_xticklines(), ax.get_yticklines()], color="white")
        ax.set_alpha(0)
        if index >= len(downsampled_img):
            ax.imshow(EMPTY_IMG, cmap=cmap)
        else:
            ax.imshow(downsampled_img[index], cmap=cmap)
    ds_len, ds_wid = downsampled_img[0].shape[0:2]
    fig.set_size_inches(width, width * ds_len / ds_wid)


def plot_histogram(
        img: npt.NDArray,
        num_bins: int = 256,
        show_img: bool = True,
        log: bool = True,
        cumulative: bool = False
):
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
    if img.shape[-1] == 3:
        return True
    else:
        return False
