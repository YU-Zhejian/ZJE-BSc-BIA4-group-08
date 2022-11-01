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
import statistics
from typing import Union, Type, List, Optional, Any, Mapping

import skimage.filters as skifilt
import skimage.exposure as skiexp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import ray
from ray.util.joblib import register_ray
import tqdm

try:
    import sklearnex

    sklearnex.patch_sklearn()
except ImportError:
    sklearnex = None

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, \
    HistGradientBoostingClassifier, BaggingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.manifold import TSNE

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
ds = covid_dataset.CovidDataSet.parallel_from_directory(os.path.join(THIS_DIR_PATH, "covid_image"))
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
_ModelTypeType = Union[
    Type[KNeighborsClassifier],
    Type[SVC],
    Type[DecisionTreeClassifier],
    Type[RandomForestClassifier],
    Type[AdaBoostClassifier],
    Type[GradientBoostingClassifier],
    Type[HistGradientBoostingClassifier],
    Type[BaggingClassifier],
    Type[VotingClassifier],
    Type[XGBClassifier],
    Type[LGBMClassifier]
]


def sklearn_get_accuracy(
        _ds: covid_dataset.CovidDataSet,
        model_type: _ModelTypeType,
        model_kwds: Optional[Mapping[str, Any]] = None
):
    if model_kwds is None:
        model_kwds = {}
    X, y = _ds.sklearn_dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True)
    model = model_type(**model_kwds)
    model = model.fit(X=X_train, y=y_train)
    pred = model.predict(X_test)
    accuracy = np.sum(pred == y_test) / len(y_test) * 100
    return accuracy


def mean_sklearn_get_accuracy(
        _ds: covid_dataset.CovidDataSet,
        model_type: _ModelTypeType,
        model_kwds: Optional[Mapping[str, Any]] = None,
        num_iter: int = 100,
        parallel: bool = True,
        backend: str = "threading"
) -> List[float]:
    def dumb_train(_):
        retv = sklearn_get_accuracy(
            _ds=_ds,
            model_type=model_type,
            model_kwds=model_kwds
        )
        gc.collect()
        return retv

    if not parallel:
        retl = list(map(
            dumb_train,
            tqdm.tqdm(iterable=range(num_iter), desc=f"Training with {model_type.__name__}...")
        ))
    else:
        retl = list(joblib_helper.parallel_map(
            dumb_train,
            tqdm.tqdm(iterable=range(num_iter), desc=f"Training with {model_type.__name__}..."),
            backend=backend
        ))
    print(f"{model_type.__name__} accuracy {statistics.mean(retl)} stdev {statistics.stdev(retl)}")
    return retl


# %%

# %% pycharm={"is_executing": true}
knn_accu = mean_sklearn_get_accuracy(ds, KNeighborsClassifier, num_iter=40)
svm_accu = mean_sklearn_get_accuracy(ds, SVC, num_iter=40)
dt_accu = mean_sklearn_get_accuracy(ds, DecisionTreeClassifier, num_iter=40)
rdn_forest_accu = mean_sklearn_get_accuracy(ds, RandomForestClassifier, num_iter=40)
adaboost_dt_accu = mean_sklearn_get_accuracy(
    ds,
    AdaBoostClassifier,
    model_kwds={"base_estimator": DecisionTreeClassifier()},
    num_iter=40,
    parallel=False
)
gbc_accu = mean_sklearn_get_accuracy(
    ds,
    GradientBoostingClassifier,
    model_kwds={"n_iter_no_change": 5, "tol": 0.01},
    num_iter=40,
    parallel=False
)
hgbc_accu = mean_sklearn_get_accuracy(
    ds,
    HistGradientBoostingClassifier,
    model_kwds={"n_iter_no_change": 5, "tol": 0.01},
    num_iter=40,
    parallel=False
)
xgb_accu = mean_sklearn_get_accuracy(ds, XGBClassifier, num_iter=40, parallel=False)
lgbm_accu = mean_sklearn_get_accuracy(ds, LGBMClassifier, num_iter=40, parallel=False)
bag_knn_accu = mean_sklearn_get_accuracy(
    ds,
    BaggingClassifier,
    model_kwds={"base_estimator": KNeighborsClassifier()},
    num_iter=40,
    parallel=False
)
bag_svm_accu = mean_sklearn_get_accuracy(
    ds,
    BaggingClassifier,
    model_kwds={"base_estimator": SVC()},
    num_iter=40,
    parallel=False
)
bag_dt_accu = mean_sklearn_get_accuracy(
    ds,
    BaggingClassifier,
    model_kwds={"base_estimator": DecisionTreeClassifier()},
    num_iter=40,
    parallel=False
)
vote_accu = mean_sklearn_get_accuracy(
    ds, VotingClassifier,
    model_kwds={"estimators": [
        ('knn', KNeighborsClassifier()),
        ('svm', SVC()),
        ('dt', DecisionTreeClassifier())
    ]},
    num_iter=40,
    parallel=False
)

# %%
acu_table = pd.DataFrame({
    "KNN": knn_accu,
    "SVM": svm_accu,
    "Decision Tree": dt_accu,
    "Random Forest": rdn_forest_accu,
    "AdaBoost (Decision Tree)": adaboost_dt_accu,
    "GBC": gbc_accu,
    "HGBC": hgbc_accu,
    "XGBoost": xgb_accu,
    "LightGBM": lgbm_accu,
    "Bagging (KNN)": bag_knn_accu,
    "Bagging (SVM)": bag_svm_accu,
    "Bagging (Decision Tree)": bag_dt_accu,
    "Voting (KNN, SVM, Decision Tree)": vote_accu
})
p = sns.catplot(data=acu_table, kind="box", height=5, aspect=2, orient="h")
p.set_axis_labels("Classification Algorithm", "Accuracy")
p.set(xlim=(10, 100))
plt.show()

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
enhanced_knn_accu = mean_sklearn_get_accuracy(ds_enhanced, KNeighborsClassifier, num_iter=40)
enhanced_svm_accu = mean_sklearn_get_accuracy(ds_enhanced, SVC, num_iter=40)
enhanced_dt_accu = mean_sklearn_get_accuracy(ds_enhanced, DecisionTreeClassifier, num_iter=40)
enhanced_rdn_forest_accu = mean_sklearn_get_accuracy(ds_enhanced, RandomForestClassifier, num_iter=40)
# adaboost_svm_accu = mean_sklearn_get_accuracy(ds, AdaBoostClassifier, model_kwds={"base_estimator":SVC(), "algorithm":'SAMME'}, num_iter=40)
enhanced_adaboost_dt_accu = mean_sklearn_get_accuracy(
    ds_enhanced,
    AdaBoostClassifier,
    model_kwds={"base_estimator": DecisionTreeClassifier()},
    num_iter=40)
enhanced_gbc_accu = mean_sklearn_get_accuracy(
    ds_enhanced,
    GradientBoostingClassifier,
    model_kwds={"n_iter_no_change": 5, "tol": 0.01},
    num_iter=40
)
enhanced_hgbc_accu = mean_sklearn_get_accuracy(
    ds_enhanced,
    HistGradientBoostingClassifier,
    model_kwds={"n_iter_no_change": 5, "tol": 0.01},
    num_iter=40
)
enhanced_xgb_accu = mean_sklearn_get_accuracy(ds_enhanced, XGBClassifier, num_iter=40)
enhanced_lgbm_accu = mean_sklearn_get_accuracy(ds_enhanced, LGBMClassifier, num_iter=40)
enhanced_bag_knn_accu = mean_sklearn_get_accuracy(
    ds_enhanced,
    BaggingClassifier,
    model_kwds={"base_estimator": KNeighborsClassifier()},
    num_iter=40
)
enhanced_bag_svm_accu = mean_sklearn_get_accuracy(
    ds_enhanced,
    BaggingClassifier,
    model_kwds={"base_estimator": SVC()},
    num_iter=40
)
enhanced_bag_dt_accu = mean_sklearn_get_accuracy(
    ds_enhanced,
    BaggingClassifier,
    model_kwds={"base_estimator": DecisionTreeClassifier()},
    num_iter=40
)
enhanced_vote_accu = mean_sklearn_get_accuracy(
    ds_enhanced,
    VotingClassifier,
    model_kwds={
        "estimators": [
            ('knn', KNeighborsClassifier()),
            ('svm', SVC()),
            ('dt', DecisionTreeClassifier())
        ]},
    num_iter=40
)

# %%
acu_table = pd.DataFrame({
    "KNN": knn_accu,
    "KNN (Post)": enhanced_knn_accu,
    "SVM": svm_accu,
    "SVM (Post)": enhanced_svm_accu,
    "Decision Tree": dt_accu,
    "Decision Tree (Post)": enhanced_dt_accu,
    "Random Forest": rdn_forest_accu,
    "Random Forest (Post)": enhanced_rdn_forest_accu,
    "AdaBoost (Decision Tree)": adaboost_dt_accu,
    "AdaBoost (Decision Tree) (Post)": enhanced_adaboost_dt_accu,
    "GBC": gbc_accu,
    "GBC (Post)": enhanced_gbc_accu,
    "HGBC": hgbc_accu,
    "HGBC (Post)": enhanced_hgbc_accu,
    "XGBoost": xgb_accu,
    "XGBoost (Post)": enhanced_xgb_accu,
    "LightGBM": lgbm_accu,
    "LightGBM (Post)": enhanced_lgbm_accu,
    "Bagging (KNN)": bag_knn_accu,
    "Bagging (KNN) (Post)": enhanced_bag_knn_accu,
    "Bagging (SVM)": bag_svm_accu,
    "Bagging (SVM) (Post)": enhanced_bag_svm_accu,
    "Bagging (Decision Tree)": bag_dt_accu,
    "Bagging (Decision Tree) (Post)": enhanced_bag_dt_accu,
    "Voting (KNN, SVM, Decision Tree)": vote_accu,
    "Voting (KNN, SVM, Decision Tree) (Post)": enhanced_vote_accu
})
p = sns.catplot(data=acu_table, kind="box", height=5, aspect=2, orient="h")
p.set_axis_labels("Classification Algorithm", "Accuracy")
p.set(xlim=(10, 100))
plt.show()

