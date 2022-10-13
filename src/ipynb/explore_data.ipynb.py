# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3.8.13 ('BIA4Env')
#     language: python
#     name: python3
# ---

# %%
# Development Block which can be safely ignored.
# %load_ext autoreload
# %autoreload 2
import os
import sys

THIS_DIR_PATH = os.path.abspath(globals()["_dh"][0])
sys.path.insert(0, os.path.dirname(THIS_DIR_PATH))

# %% [markdown]
# # Examples of BIA_KiTS19 Data Pre-Processing Infrastructure
#
# Here contains some examples on how to use the infrastructure.
#
# We would first import important libraries.

# %%
# Import some SKLearn acceleration libraries
try:
    import sklearnex

    sklearnex.patch_sklearn()
except ImportError:
    sklearnex = None

import numpy as np
import skimage
import skimage.exposure as skiexp
import skimage.filters as skifilt

# Import the infrastructure
from BIA_KiTS19.helper import dataset_helper
from BIA_KiTS19.helper import matplotlib_helper
from BIA_KiTS19.helper import skimage_helper

# %% [markdown]
# ## Loading DataSet
#
# The `dataset_helper.DataSet` infrastructure automatically loads, processes and caches CT data in NIFTI into images stored in 3d `npt.NDArray[float]`.

# %%
dataset = dataset_helper.DataSet("/media/yuzj/BUP/kits19/data")

# %% [markdown]
# Print the name of top 10 dataset. The `dataset.iter_case_names()` method would create an iterable which iterates case names. For example:
#
# ```python
# from typing import List
# from BIA_KiTS19.helper import dataset_helper
#
# dataset: dataset_helper.DataSet = None
# dataset_case_names: List[str] = list(dataset.iter_case_names())
# dataset_image_sets: List[dataset_helper.ImageSet] = list(dataset)
# ```
#
# would load all `dataset_helper.ImageSet` inside the dataset.

# %%
list(dataset.iter_case_names())[0:10]

# %% [markdown]
# Noe we load the first dataset.

# %%
case1 = dataset["/media/yuzj/BUP/kits19/data/case_00001/"]

# %% [markdown]
# Pre-processed image ikn numpy format would be automatically loaded (or processed and loaded) using `np_image_final` and `np_mask_final` properties.

# %%
case1_img, case1_mask = case1.np_image_final, case1.np_mask_final

# %%
case1_img.shape, case1_mask.shape

# %% [markdown]
# Use `matplotlib_helper.plot_3d_rgba` to plot several slices of the image on some axis.

# %%
matplotlib_helper.plot_3d_rgba(case1_img, axis=1)

# %%
matplotlib_helper.plot_3d_rgba(case1_mask, axis=0)

# %% [markdown]
# Add masks to images and plot them.

# %%
case1_rgba = skimage_helper.image_mask_to_rgba(case1_img, case1_mask)
matplotlib_helper.plot_3d_rgba(case1_rgba, axis=1, num_slices=12)

# %%
matplotlib_helper.plot_histogram(img=case1_img, show_img=False, log=False, cumulative=True)

# %%
case1_img_stretch = skiexp.rescale_intensity(
    case1_img,
    in_range=tuple(np.percentile(case1_img, (2, 98))),
    out_range=(0, 255)
)

# %%
matplotlib_helper.plot_histogram(img=case1_img_stretch, show_img=False, log=False, cumulative=True)

# %%
matplotlib_helper.plot_3d_rgba(case1_img_stretch, axis=1, num_slices=12)

# %%
case1_stretch_rgba = skimage_helper.image_mask_to_rgba(case1_img_stretch, case1_mask)
matplotlib_helper.plot_3d_rgba(case1_stretch_rgba, axis=1, num_slices=12)

# %%
case1_img_equalized = skimage.img_as_uint(skiexp.equalize_hist(case1_img))

# %%
matplotlib_helper.plot_histogram(img=case1_img_equalized, show_img=False, log=False, cumulative=True)

# %%
matplotlib_helper.plot_3d_rgba(case1_img_equalized, axis=1, num_slices=12)

# %%
case1_img_equalized_rgba = skimage_helper.image_mask_to_rgba(case1_img_equalized, case1_mask)
matplotlib_helper.plot_3d_rgba(case1_img_equalized_rgba, axis=1, num_slices=12)

# %%
case1_img_unsharp = skifilt.unsharp_mask(case1_img, radius=5, amount=2)
matplotlib_helper.plot_3d_rgba(case1_img_unsharp, axis=1, num_slices=12)

# %%
