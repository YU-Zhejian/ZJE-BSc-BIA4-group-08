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
# %matplotlib inline

import os
import sys

try:
    THIS_DIR_PATH = os.path.abspath(globals()["_dh"][0])
except KeyError:
    THIS_DIR_PATH = os.path.dirname(os.path.abspath(__file__))
NEW_PYTHON_PATH = os.path.join(os.path.dirname(os.path.dirname(THIS_DIR_PATH)), "src")
sys.path.insert(0, NEW_PYTHON_PATH)
os.environ["PYTHONPATH"] = os.pathsep.join((NEW_PYTHON_PATH, os.environ.get("PYTHONPATH", "")))

# %% [markdown]
# # Machine-Learning Optimization: A Failed Example
#
# Here we would provide a failed attempt to optimize machine-learning results. It would use the COVID dataset infrastructure on real pictures.
#
# Before starting, we would import necessary libraries.

# %%
import gc  # For collecting memory garbage

import skimage.filters as skifilt
import skimage.exposure as skiexp
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import joblib
import ray
from ray.util.joblib import register_ray

try:
    import sklearnex

    sklearnex.patch_sklearn()
except ImportError:
    sklearnex = None

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as KNN

from BIA_G8.covid_helper import covid_dataset
from BIA_G8.helper import ml_helper, matplotlib_helper

# %% [markdown]
# Start local `ray` server.

# %%
if not ray.is_initialized():
    ray.init()
register_ray()

# %% [markdown]
# Read and downscale the dataset.

# %%
ds = covid_dataset.CovidDataSet.parallel_from_directory(os.path.join(THIS_DIR_PATH, "sample_covid_image"))
_ = gc.collect()

# %% [markdown]
# Use `sklearn` on this raw dataset.

# %%
ds_sklearn = ds.sklearn_dataset

accuracy = []
with joblib.parallel_backend('ray'):
    for _ in tqdm.tqdm(iterable=range(400)):
        X, y = ds_sklearn
        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True)
        accuracy.append(np.sum(KNN().fit(X=X_train, y=y_train).predict(X_test) == y_test) / len(y_test) * 100)
print(f"Accuracy: {np.mean(accuracy):4.2f}")


# %% [markdown]
# ## Optimization using Histogram Equalization and Unsharp Masking

# %%
fig, axs = plt.subplots(3, 3, figsize=(12, 12))

sample_figures = ds.sample(9)

ax: plt.Axes
for i, ax in enumerate(axs.ravel()):
    ax.imshow(sample_figures[i].np_array)
    ax.axis("off")
    ax.set_title(sample_figures[i].label_str)

# %%
matplotlib_helper.plot_histogram(ds[0].np_array, cumulative=True, log=False)

# %%
ds_equalize_hist = ds.parallel_apply(
    skiexp.equalize_hist,
    backend="threading"
)

# %%
matplotlib_helper.plot_histogram(ds_equalize_hist[0].np_array, cumulative=True, log=False)

# %%
ds_enhanced = ds_equalize_hist.parallel_apply(
    lambda img: skifilt.unsharp_mask(img, radius=5, amount=3),
    backend="threading"
)

# %%
fig, axs = plt.subplots(3, 3, figsize=(12, 12))

sample_figures = ds_enhanced.sample(9)

ax: plt.Axes
for i, ax in enumerate(axs.ravel()):
    ax.imshow(sample_figures[i].np_array)
    ax.axis("off")
    ax.set_title(sample_figures[i].label_str)

# %% [markdown]
# ## Re-learn using SKLearn KNN

# %%
accuracy = []
with joblib.parallel_backend('ray'):
    for _ in tqdm.tqdm(iterable=range(400)):
        X, y = ds_sklearn
        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True)
        accuracy.append(np.sum(KNN().fit(X=X_train, y=y_train).predict(X_test) == y_test) / len(y_test) * 100)
print(f"Accuracy: {np.mean(accuracy):4.2f}")
