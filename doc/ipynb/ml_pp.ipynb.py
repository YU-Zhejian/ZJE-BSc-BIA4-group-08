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
import warnings

warnings.filterwarnings('ignore')
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
import skimage.filters as skifilt
import skimage.exposure as skiexp

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from BIA_G8.data_analysis import covid_dataset
from BIA_G8.helper import matplotlib_helper

# %% [markdown]
# Read and downscale the dataset.

# %%
ds = covid_dataset.CovidDataSet.parallel_from_directory(os.path.join(THIS_DIR_PATH, "sample_covid_image"))

# %% [markdown]
# Use `sklearn` on this raw dataset.

# %%
accuracy = []
for _ in tqdm.tqdm(iterable=range(200)):
    X, y = ds.sklearn_dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True)
    accuracy.append(
        np.sum(KNeighborsClassifier().fit(X=X_train, y=y_train).predict(X_test) == y_test) / len(y_test) * 100)
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
    skiexp.equalize_hist
)

# %%
matplotlib_helper.plot_histogram(ds_equalize_hist[0].np_array, cumulative=True, log=False)

# %%
ds_enhanced = ds_equalize_hist.parallel_apply(
    lambda img: skifilt.unsharp_mask(img, radius=5, amount=3)
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
accuracy_new = []
for _ in tqdm.tqdm(iterable=range(200)):
    X, y = ds_enhanced.sklearn_dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True)
    accuracy_new.append(
        np.sum(KNeighborsClassifier().fit(X=X_train, y=y_train).predict(X_test) == y_test) / len(y_test) * 100)
print(f"Accuracy: {np.mean(accuracy_new):4.2f}")

# %% [markdown]
# plot them

# %%
acu_table = pd.DataFrame({
    "KNN (Pre)": accuracy,
    "KNN (Post)": accuracy_new,
})
p = sns.catplot(data=acu_table, kind="box", height=5, aspect=2, orient="h")
p.set_axis_labels("Accuracy", "Classification Algorithm")
p.set(xlim=(10, 100))
plt.show()
