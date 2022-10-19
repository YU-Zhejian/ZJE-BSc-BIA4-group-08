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

THIS_DIR_PATH = os.path.abspath(globals()["_dh"][0])
NEW_PYTHON_PATH = os.path.dirname(THIS_DIR_PATH)
sys.path.insert(0, NEW_PYTHON_PATH)
os.environ["PYTHONPATH"] = os.sep.join((NEW_PYTHON_PATH, os.environ.get("PYTHONPATH", "")))

# %%
import random

import skimage
import skimage.transform as skitrans
import skimage.draw as skidraw
import skimage.util as skiutil
import skimage.color as skicol
import skimage.filters as skifilt
import skimage.exposure as skiexp
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
from sklearn.ensemble import GradientBoostingClassifier as GBC
from xgboost import XGBClassifier
from lightgbm.sklearn import LGBMClassifier

from BIA_G8.covid_helper import covid_dataset
from BIA_G8.helper import ml_helper, matplotlib_helper

# %%
if not ray.is_initialized():
    ray.init()
register_ray()

# %%
ds = covid_dataset.CovidDataSet.parallel_from_directory("/media/yuzj/BUP/covid19-database-np", size=120)

# %%
ds_sklearn = ds.get_sklearn_dataset

with joblib.parallel_backend('ray'):
    X, y = ds_sklearn
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True)
    knn = KNN()
    knn = knn.fit(X=X_train, y=y_train)
    pred = knn.predict(X_test)
    accuracy = np.sum(pred == y_test) / len(y_test)*100
    print(f"Accuracy: {accuracy:4.2f}")
    _confusion_matrix = confusion_matrix(pred, y_test)
    print(ml_helper.print_confusion_matrix(_confusion_matrix, labels=["stride", "circle", "square"]))

# %%
fig, axs = plt.subplots(3, 3, figsize=(12, 12))

sample_figures = ds.sample(9)

ax: plt.Axes
for i, ax in enumerate(axs.ravel()):
    ax.imshow(sample_figures[i].as_np_array)
    ax.axis("off")
    ax.set_title(sample_figures[i].label_str)

# %%
matplotlib_helper.plot_histogram(ds[0].as_np_array, cumulative=True, log=False)

# %%
ds_equalize_hist = ds.parallel_apply(
    skiexp.equalize_hist,
    backend="threading"
)

# %%
matplotlib_helper.plot_histogram(ds_equalize_hist[0].as_np_array, cumulative=True, log=False)

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
    ax.imshow(sample_figures[i].as_np_array)
    ax.axis("off")
    ax.set_title(sample_figures[i].label_str)

# %%
with joblib.parallel_backend('ray'):
    X, y = ds_enhanced.get_sklearn_dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True)
    knn = KNN()
    knn = knn.fit(X=X_train, y=y_train)
    pred = knn.predict(X_test)
    accuracy = np.sum(pred == y_test) / len(y_test)*100
    print(f"Accuracy: {accuracy:4.2f}")
    _confusion_matrix = confusion_matrix(pred, y_test)
    print(ml_helper.print_confusion_matrix(_confusion_matrix, labels=["stride", "circle", "square"]))

# %%
X, y = ds_enhanced.get_sklearn_dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True)
xgbc = XGBClassifier(verbosity=3)
xgbc = xgbc.fit(X=X_train, y=y_train)
pred = xgbc.predict(X_test)
accuracy = np.sum(pred == y_test) / len(y_test)*100
print(f"Accuracy: {accuracy:4.2f}")
_confusion_matrix = confusion_matrix(pred, y_test)
print(ml_helper.print_confusion_matrix(_confusion_matrix, labels=["stride", "circle", "square"]))

# %%
X, y = ds.get_sklearn_dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True)
xgbc = XGBClassifier(verbosity=3)
xgbc = xgbc.fit(X=X_train, y=y_train)
pred = xgbc.predict(X_test)
accuracy = np.sum(pred == y_test) / len(y_test)*100
print(f"Accuracy: {accuracy:4.2f}")
_confusion_matrix = confusion_matrix(pred, y_test)
print(ml_helper.print_confusion_matrix(_confusion_matrix, labels=["stride", "circle", "square"]))

# %%
X, y = ds_enhanced.get_sklearn_dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True)
lgbc = LGBMClassifier(n_jobs=-1, verbose=2)
lgbc = lgbc.fit(X=X_train, y=y_train)
pred = lgbc.predict(X_test)
accuracy = np.sum(pred == y_test) / len(y_test)*100
print(f"Accuracy: {accuracy:4.2f}")
_confusion_matrix = confusion_matrix(pred, y_test)
print(ml_helper.print_confusion_matrix(_confusion_matrix, labels=["stride", "circle", "square"]))

# %%
X, y = ds.get_sklearn_dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True)
lgbc = LGBMClassifier(n_jobs=-1, verbose=2)
lgbc = lgbc.fit(X=X_train, y=y_train)
pred = lgbc.predict(X_test)
accuracy = np.sum(pred == y_test) / len(y_test)*100
print(f"Accuracy: {accuracy:4.2f}")
_confusion_matrix = confusion_matrix(pred, y_test)
print(ml_helper.print_confusion_matrix(_confusion_matrix, labels=["stride", "circle", "square"]))
