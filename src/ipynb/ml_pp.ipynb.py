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
os.environ["PYTHONPATH"] = os.pathsep.join((NEW_PYTHON_PATH, os.environ.get("PYTHONPATH", "")))

# %% [markdown]
# # Machine-Learning Optimization: A Failed Example
#
# Here we would provide a failed attempt to optimize machine-learning results. It would use the COVID dataset infrastructure on real pictures.
#
# Before starting, we would import necessary libraries.

# %%
import gc  # For collecting memory garbage
from typing import Union, Type, List

import skimage.filters as skifilt
import skimage.exposure as skiexp
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import joblib
import ray
from ray.util.joblib import register_ray
import tqdm

try:
    import sklearnex

    sklearnex.patch_sklearn()
except ImportError:
    sklearnex = None

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.svm import SVC as SVM
from sklearn.manifold import TSNE
import seaborn as sns
from xgboost import XGBClassifier
from lightgbm.sklearn import LGBMClassifier

from BIA_G8.covid_helper import covid_dataset
from BIA_G8.helper import matplotlib_helper, joblib_helper

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
ds_sklearn = ds.sklearn_dataset
_ = gc.collect()

# %% [markdown]
# Plot t-SNE of the data.

# %%
with joblib.parallel_backend('ray'):
    ds_tsne_model = TSNE(learning_rate=200, n_iter=1000, init="random")
    ds_tsne_transformed = ds_tsne_model.fit_transform(ds_sklearn[0])
    sns.scatterplot(
        x=ds_tsne_transformed[:, 0],
        y=ds_tsne_transformed[:, 1],
        hue=list(map(covid_dataset.decode, ds_sklearn[1]))
    )


# %% [markdown]
# Use `sklearn` on this raw dataset.

# %%
def sklearn_get_accuracy(
    _ds:npt.NDArray,
    ModelType:Union[
        Type[KNN],
        Type[SVM],
        Type[XGBClassifier],
        Type[LGBMClassifier]
    ]
):
    X, y = ds_sklearn
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True)
    model = ModelType()
    model = model.fit(X=X_train, y=y_train)
    pred = model.predict(X_test)
    accuracy = np.sum(pred == y_test) / len(y_test) * 100
    return accuracy

def parallel_mean_sklearn_get_accuracy(
    _ds:npt.NDArray,
    ModelType:Union[Type[KNN], Type[SVM], Type[XGBClassifier], Type[LGBMClassifier]]
) -> List[float]:
    return list(joblib_helper.parallel_map(
        lambda _: sklearn_get_accuracy(
            _ds=_ds,
            ModelType=ModelType
        ),
        tqdm.tqdm(iterable=range(100), desc="Training...")
    ))

np.mean(parallel_mean_sklearn_get_accuracy(ds, KNN))

# %%

# %%
X, y = ds.sklearn_dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True)
xgbc = XGBClassifier()
xgbc = xgbc.fit(X=X_train, y=y_train)
pred = xgbc.predict(X_test)
accuracy = np.sum(pred == y_test) / len(y_test) * 100
print(f"Accuracy: {accuracy:4.2f}")
_confusion_matrix = confusion_matrix(pred, y_test)
print(ml_helper.print_confusion_matrix(_confusion_matrix, labels=["stride", "circle", "square"]))

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
# ## Re-learn using SKLearn KNN and two GBT-Based Algorithm

# %%
with joblib.parallel_backend('ray'):
    X, y = ds_enhanced.sklearn_dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True)
    knn = KNN()
    knn = knn.fit(X=X_train, y=y_train)
    pred = knn.predict(X_test)
    accuracy = np.sum(pred == y_test) / len(y_test) * 100
    print(f"Accuracy: {accuracy:4.2f}")
    _confusion_matrix = confusion_matrix(pred, y_test)
    print(ml_helper.print_confusion_matrix(_confusion_matrix, labels=["stride", "circle", "square"]))

# %% notebookRunGroups={"groupValue": ""}
X, y = ds_enhanced.sklearn_dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True)
xgbc = XGBClassifier()
xgbc = xgbc.fit(X=X_train, y=y_train)
pred = xgbc.predict(X_test)
accuracy = np.sum(pred == y_test) / len(y_test) * 100
print(f"Accuracy: {accuracy:4.2f}")
_confusion_matrix = confusion_matrix(pred, y_test)
print(ml_helper.print_confusion_matrix(_confusion_matrix, labels=["stride", "circle", "square"]))

# %%
X, y = ds.sklearn_dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True)
lgbc = LGBMClassifier(n_jobs=-1)
lgbc = lgbc.fit(X=X_train, y=y_train)
pred = lgbc.predict(X_test)
accuracy = np.sum(pred == y_test) / len(y_test) * 100
print(f"Accuracy: {accuracy:4.2f}")
_confusion_matrix = confusion_matrix(pred, y_test)
print(ml_helper.print_confusion_matrix(_confusion_matrix, labels=["stride", "circle", "square"]))

# %%
X, y = ds_enhanced.sklearn_dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True)
lgbc = LGBMClassifier(n_jobs=-1)
lgbc = lgbc.fit(X=X_train, y=y_train)
pred = lgbc.predict(X_test)
accuracy = np.sum(pred == y_test) / len(y_test) * 100
print(f"Accuracy: {accuracy:4.2f}")
_confusion_matrix = confusion_matrix(pred, y_test)
print(ml_helper.print_confusion_matrix(_confusion_matrix, labels=["stride", "circle", "square"]))
