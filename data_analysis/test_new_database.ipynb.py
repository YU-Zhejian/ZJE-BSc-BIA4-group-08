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
NEW_PYTHON_PATH = os.path.join(os.path.dirname(THIS_DIR_PATH), "src")
sys.path.insert(0, NEW_PYTHON_PATH)
os.environ["PYTHONPATH"] = os.pathsep.join((NEW_PYTHON_PATH, os.environ.get("PYTHONPATH", "")))

# %%
import gc  # For collecting memory garbage
import numpy as np
import pandas as pd
import skimage.transform as skitrans

import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE

from BIA_G8_DATA_ANALYSIS import covid_dataset
from BIA_G8.helper import ml_helper

# %%
encode, decode = ml_helper.generate_encoder_decoder({
    "NA": 100,
    "COVID": 0,
    "Lung_Opacity": 1,
    "Normal": 2,
    "Viral Pneumonia": 3,
})

# %%
ds = covid_dataset.CovidDataSet.parallel_from_directory(
    os.path.join(THIS_DIR_PATH, "covid_image_new"),
    size=400,
    encode=encode, decode=decode
).parallel_apply(
    lambda img: skitrans.resize(
        img,
        (128, 128)
    )
)
_ = gc.collect()

# %%
fig, axs = plt.subplots(3, 3, figsize=(12, 12))

sample_figures = ds.sample(9)

ax: plt.Axes
for i, ax in enumerate(axs.ravel()):
    ax.imshow(sample_figures[i].np_array)
    ax.axis("off")
    ax.set_title(sample_figures[i].label_str)

# %%
ds_tsne_transformed = TSNE(learning_rate=200, n_iter=1000, init="random").fit_transform(ds.sklearn_dataset[0])
sns.scatterplot(
    x=ds_tsne_transformed[:, 0],
    y=ds_tsne_transformed[:, 1],
    hue=list(map(ds.decode, ds.sklearn_dataset[1]))
)
plt.show()
del ds_tsne_transformed
_ = gc.collect()

# %%
accuracy = []
for _ in tqdm(iterable=range(200)):
    X, y = ds.sklearn_dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True)
    accuracy.append(
        np.sum(KNeighborsClassifier().fit(X=X_train, y=y_train).predict(X_test) == y_test) / len(y_test) * 100)
print(f"Accuracy: {np.mean(accuracy):4.2f}")

# %%
import skimage.exposure as skiexp
import numpy.typing as npt


def adaphist(img: npt.NDArray) -> npt.NDArray:
    q2, q95 = np.percentile(img, (2, 95))
    return skiexp.equalize_adapthist(skiexp.rescale_intensity(img, in_range=(q2, q95), out_range=(-1, 1)))


ds_equalize_hist = ds.parallel_apply(
    adaphist
)

# %%
accuracy_new = []
for _ in tqdm(iterable=range(200)):
    X, y = ds_equalize_hist.sklearn_dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True)
    accuracy_new.append(
        np.sum(KNeighborsClassifier().fit(X=X_train, y=y_train).predict(X_test) == y_test) / len(y_test) * 100)
print(f"Accuracy: {np.mean(accuracy_new):4.2f}")

# %%
acu_table = pd.DataFrame({
    "KNN (Pre)": accuracy,
    "KNN (Post)": accuracy_new,
})
p = sns.catplot(data=acu_table, kind="box", height=5, aspect=2, orient="h")
p.set_axis_labels("Accuracy", "Classification Algorithm")
p.set(xlim=(10, 100))
plt.show()
