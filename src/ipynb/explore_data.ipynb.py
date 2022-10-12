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
# %load_ext autoreload
# %autoreload 2
import os
import sys

THIS_DIR_PATH = os.path.abspath(globals()["_dh"][0])
sys.path.insert(0, os.path.dirname(THIS_DIR_PATH))

# %%
try:
    import sklearnex

    sklearnex.patch_sklearn()
except ImportError:
    pass

import numpy as np
import skimage
import skimage.exposure as skiexp
import skimage.filters as skifilt

from BIA_KiTS19.helper import dataset_helper
from BIA_KiTS19.helper import matplotlib_helper
from BIA_KiTS19.helper import skimage_helper

# %%
dataset = dataset_helper.DataSet("/media/yuzj/BUP/kits19/data")

# %%
list(dataset.iter_case_names())[0:10]

# %%
case1 = dataset["/media/yuzj/BUP/kits19/data/case_00001/"]

# %%
case1_img, case1_mask = case1.space_resampled_np_image, case1.space_resampled_np_mask

# %%
case1_img.shape, case1_mask.shape

# %%
matplotlib_helper.plot_3d_rgba(case1_img, axis=1)

# %%
matplotlib_helper.plot_3d_rgba(case1_mask, axis=0)

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
