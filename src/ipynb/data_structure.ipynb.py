# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3
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

THIS_DIR_PATH = os.path.abspath(globals()["_dh"][0])
sys.path.insert(0, os.path.dirname(THIS_DIR_PATH))

# %% [markdown]
# # Fast In-Memory Analytics using COVID Data Infrastructure
#
# In this example, we would show how to use the COVID data infrastructure to accomplish following tasks:
#
# - Explore the dataset.
# - Apply a function to the dataset.
# - Perform machine-learning on the dataset.
# - Save the dataset.
#
# Before we start, we would import several Python packages. They are mainly:
#
# - `skimage`, which provides image transformation functions;
# - `operator`, `itertools` and `functools`, which provides functional programming (FP) primitives;
# - `ray` and `joblib`, which provides parallel support;
# - `sklearn`, the simpliest machine-learning package.

# %%
import skimage
import skimage.transform as skitrans
import skimage.draw as skidraw
import skimage.util as skiutil
import numpy as np
import operator
from functools import reduce
import matplotlib.pyplot as plt
import itertools
import joblib
import ray
from ray.util.joblib import register_ray
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as KNN

from BIA_G8.covid_helper import covid_dataset

if not ray.is_initialized():
    ray.init()
register_ray()

# %% [markdown]
# ## Generate Stride Images
#
# Here we would generate some image of strides, circles and squares as our initial dataset.

# %%
blank = np.zeros((100, 100), dtype=int)
stride = skimage.img_as_int(np.array(reduce(operator.add, map(lambda x: [[x] * 100] * 10, range(0, 100, 10)))))

circle = blank.copy()
rr, cc = skidraw.circle_perimeter(r=50, c=50, radius=30)
circle[rr, cc] = 100
circle = skimage.img_as_int(circle)

square = blank.copy()
rr, cc = skidraw.rectangle(start=(20, 20), extent=(60, 60))
square[rr, cc] = 100
square = skimage.img_as_int(square)

_IMAGES = [
    covid_dataset.CovidImage.from_image(stride, 0),
    covid_dataset.CovidImage.from_image(stride, 0),
    covid_dataset.CovidImage.from_image(circle, 1),
    covid_dataset.CovidImage.from_image(circle, 1),
    covid_dataset.CovidImage.from_image(square, 2),
    covid_dataset.CovidImage.from_image(square, 2),
]
d1 = covid_dataset.CovidDataSet.from_loaded_image(_IMAGES)

# %% [markdown]
# ## Sample and Plot
#
# Now we generate a random sample consisting of 3 images. The `balanced` argument indicate whether the cases from different categories should be same.

# %%
d1_sampled = d1.sample(3, balanced=True)

# %% [markdown]
# Plot them.

# %%
fig, axs = plt.subplots(1, 3)

ax: plt.Axes
for i, ax in enumerate(axs.ravel()):
    ax.imshow(d1_sampled[i].as_np_array)
    ax.axis("off")
    ax.set_title(d1_sampled[i].label_str)

# %% [markdown]
# ## Map-like `apply` Function
#
# The `apply` function is like `map` function of Python or `apply` function in R. It applies a function over all images and generates a new dataset.
#
# For example, here we would rotate all sampled images:

# %%
d1_sampled_applied = d1_sampled.apply(lambda x: skitrans.rotate(x, 30))

# %%
fig, axs = plt.subplots(1, 3)

for i, ax in enumerate(axs.ravel()):
    ax.imshow(d1_sampled_applied[i].as_np_array)
    ax.axis("off")
    ax.set_title(d1_sampled[i].label_str)

# %% [markdown]
# ## Machine-Learning Using SKLearn
#

# %%
ds_enlarged = covid_dataset.CovidDataSet.from_loaded_image(
    list(itertools.chain(*itertools.repeat(list(d1), 200)))
).apply(
    lambda img: skimage.img_as_int(
        skiutil.random_noise(
            skimage.img_as_float(img), mode="pepper"
        )
    )
)
ds_enlarged_sampled = ds_enlarged.sample(9)

# %%
fig, axs = plt.subplots(3, 3, figsize=(12, 12))

for i, ax in enumerate(axs.ravel()):
    ax.imshow(ds_enlarged_sampled[i].as_np_array)
    ax.axis("off")
    ax.set_title(ds_enlarged_sampled[i].label_str)

# %%
ds_sklearn = ds_enlarged.get_sklearn_dataset

# %%
with joblib.parallel_backend('ray'):
    X, y = ds_sklearn
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True)
    knn = KNN()
    knn = knn.fit(X=X_train, y=y_train)
    pred = knn.predict(X_test)
    accuracy = np.sum(pred == y_test) / len(y_test)
    print(accuracy)
    _confusion_matrix = confusion_matrix(pred, y_test)
    print(_confusion_matrix)

# %%
