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

# pylint: disable=wrong-import-position, line-too-long, missing-module-docstring

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
# # Fast In-Memory Analytics using COVID Data Infrastructure
#
# The COVID data infrastructure is a dataframe-like data loader optimized for fast **interactive** in-memory analytics of small datasets.
#
# In this example, we would show how to use the COVID data infrastructure to accomplish following tasks:
#
# - Explore the dataset.
# - Apply a function to the dataset.
# - Perform machine-learning (classification using KNN) on the dataset.
# - Save the dataset.
#
# Before we start, we would import several Python packages. They are mainly:
#
# - `skimage`, which provides image transformation functions;
# - `operator`, `itertools` and `functools`, which provides functional programming (FP) primitives;
# - `joblib`, which provides parallel support;
# - `sklearn`, the simpliest machine-learning package.

# %%
import random
import operator
from functools import reduce
import itertools

import skimage
import skimage.transform as skitrans
import skimage.draw as skidraw
import skimage.util as skiutil
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as KNN

from BIA_G8_DATA_ANALYSIS import covid_dataset
from BIA_G8.helper import ml_helper

# %% [markdown]
# ## Generate Stride Images
#
# Here we would generate some image of strides, circles and squares as our initial dataset.

# %%
labels = ["stride", "circle", "square"]
_encoder_dict = {k: v for k, v in zip(labels, range(len(labels)))}
encoder, decoder = ml_helper.generate_encoder_decoder(_encoder_dict)

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

# %% [markdown]
# Display sample data.

# %%
fig, axs = plt.subplots(1, 3, figsize=(12, 12))

sample_figures = (stride, circle, square)

ax: plt.Axes
for i, ax in enumerate(axs.ravel()):
    ax.imshow(sample_figures[i], cmap="bone")
    ax.axis("off")

# %% [markdown]
# Create a dataset with 2 strides, 2 circles and 2 squares.

# %%
_IMAGES = []
for label, img in enumerate(sample_figures):
    _IMAGES.append(covid_dataset.CovidImage.from_np_array(img, label, decoder(label)))
    _IMAGES.append(covid_dataset.CovidImage.from_np_array(img, label, decoder(label)))

d1 = covid_dataset.CovidDataSet.from_loaded_image(_IMAGES, encode=encoder, decode=decoder)

# %% [markdown]
# ## Sample and Plot
#
# Now we generate a random sample consisting of 3 images. The `balanced` argument indicate whether the cases from different categories should be same.

# %%
d1_sampled = d1.sample(3, balanced=True)

# %% [markdown]
# Plot them.

# %%
fig, axs = plt.subplots(1, 3, figsize=(12, 12))

ax: plt.Axes
for i, ax in enumerate(axs.ravel()):
    ax.imshow(d1_sampled[i].np_array, cmap="bone")
    ax.axis("off")
    ax.set_title(d1_sampled[i].label_str)

# %% [markdown]
# We may also perform unbalanced sampling.

# %%
d1_sampled = d1.sample(4, balanced=False)

fig, axs = plt.subplots(1, len(d1_sampled), figsize=(12, 12))

ax: plt.Axes
for i, ax in enumerate(axs.ravel()):
    ax.imshow(d1_sampled[i].np_array, cmap="bone")
    ax.axis("off")
    ax.set_title(d1_sampled[i].label_str)

# %% [markdown]
# To make the task more complicate, we will now enlarge the dataset to 1200 cases. The variable `ds_enlarged` would contain 1200 cases and how it was generated does not need to be understood.

# %%
ds_enlarged = covid_dataset.CovidDataSet.from_loaded_image(
    list(itertools.chain(*itertools.repeat(list(d1), 200))),
    encode=d1.encode,
    decode=d1.decode
)

# %%
print(len(ds_enlarged))

# %% [markdown]
# ## Map-like `apply` Function
#
# The `apply` function is like `map` function of Python or `apply` function in R. It applies a function over all images and generates a new dataset.
#
# For example, here we would rotate all sampled images:

# %%
d1_sampled_applied = d1.sample(3).apply(lambda x: skitrans.rotate(x, 30))

# %% [markdown]
# Plot them.

# %%
fig, axs = plt.subplots(1, 3)

for i, ax in enumerate(axs.ravel()):
    ax.imshow(d1_sampled_applied[i].np_array, cmap="bone")
    ax.axis("off")
    ax.set_title(d1_sampled_applied[i].label_str)

# %% [markdown]
# The `apply` function also have its parallel function, `parallel_apply`. The following example adds noise to enlarged dataset:

# %%
ds_enlarged_with_noise = ds_enlarged.parallel_apply(
    lambda img: skimage.img_as_int(
        skiutil.random_noise(
            skimage.img_as_float(img),
            mode="pepper"
        )
    )
).parallel_apply(
    lambda img: skimage.img_as_int(
        skitrans.rotate(
            img,
            random.random() * 120 - 60
        )
    )
)

# %% [markdown]
# ## Machine-Learning Using SKLearn
#

# %% [markdown]
# Before applying ML algorithms, we would plot some example dataset:

# %%
ds_enlarged_sampled = ds_enlarged_with_noise.sample(9)

fig, axs = plt.subplots(3, 3, figsize=(12, 12))

for i, ax in enumerate(axs.ravel()):
    ax.imshow(ds_enlarged_sampled[i].np_array, cmap="bone")
    ax.axis("off")
    ax.set_title(ds_enlarged_sampled[i].label_str)

# %% [markdown]
# Convert the dataset to SKLearn-acceptable format:

# %%
ds_sklearn = ds_enlarged_with_noise.sklearn_dataset

# %% [markdown]
# Apply ML algorithms using KNN:

# %%
X, y = ds_sklearn
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True)
knn = KNN()
knn = knn.fit(X=X_train, y=y_train)
pred = knn.predict(X_test)
accuracy = np.sum(pred == y_test) / len(y_test)
print(accuracy)
_confusion_matrix = confusion_matrix(pred, y_test)
print(ml_helper.print_confusion_matrix(_confusion_matrix, labels=labels))

# %% [markdown]
# The accuracy is 100%, which is good.
#
