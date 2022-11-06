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

# %% [markdown]
# # Screening General-Purposed Unoptimized Machine-Learning Algorithms
#
# TODO
#
# Before starting, we would import necessary libraries.

# %%
import gc  # For collecting memory garbage
import statistics
from typing import Union, Type, List, Optional, Any, Mapping
import multiprocessing

import numpy as np
import pandas as pd
import skimage.transform as skitrans

import matplotlib.pyplot as plt
import seaborn as sns

from tqdm.notebook import tqdm

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, \
    HistGradientBoostingClassifier, BaggingClassifier, VotingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier, RidgeClassifier, PassiveAggressiveClassifier
from sklearn.manifold import TSNE

from xgboost.sklearn import XGBClassifier

from BIA_G8.covid_helper import covid_dataset
from BIA_G8.helper import joblib_helper

# %% [markdown]
# Read and downscale the dataset.

# %%
ds = covid_dataset.CovidDataSet.parallel_from_directory(
    os.path.join(THIS_DIR_PATH, "covid_image"),
    size=300
).parallel_apply(
    lambda img: skitrans.resize(
        img,
        (128, 128)
    )
)
_ = gc.collect()

# %% [markdown]
# Plot and t-SNE of the data.

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

# %% [markdown]
# Use `sklearn` on this raw dataset.

# %%
_ModelTypeType = Union[
    Type[KNeighborsClassifier],
    Type[MLPClassifier],
    Type[SGDClassifier],
    Type[RidgeClassifier],
    Type[PassiveAggressiveClassifier],
    Type[SVC],
    Type[DecisionTreeClassifier],
    Type[RandomForestClassifier],
    Type[AdaBoostClassifier],
    Type[GradientBoostingClassifier],
    Type[HistGradientBoostingClassifier],
    Type[ExtraTreesClassifier],
    Type[BaggingClassifier],
    Type[VotingClassifier],
    Type[XGBClassifier]
]


def sklearn_get_accuracy(
        _ds: covid_dataset.CovidDataSet,
        model_type: _ModelTypeType,
        model_kwds: Mapping[str, Any]
):
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
        num_iter: int = 10,
        parallel: bool = False,
        joblib_kwds: Optional[Mapping[str, Any]] = None,
) -> List[float]:
    def dumb_train(_):
        retv = sklearn_get_accuracy(
            _ds=_ds,
            model_type=model_type,
            model_kwds=model_kwds
        )
        gc.collect()
        return retv

    if joblib_kwds is None:
        joblib_kwds = {}
    if model_kwds is None:
        model_kwds = {}
    model_name = model_type.__name__
    if "base_estimator" in model_kwds:
        model_name += f" ({model_kwds['base_estimator'].__class__.__name__})"
    if parallel:
        retl = list(joblib_helper.parallel_map(
            dumb_train,
            tqdm(iterable=range(num_iter), desc=f"Training with {model_name}..."),
            **joblib_kwds
        ))
    else:
        retl = list(map(
            dumb_train,
            tqdm(iterable=range(num_iter), desc=f"Training with {model_name}...")
        ))
    print(f"{model_name} accuracy {statistics.mean(retl)} stdev {statistics.stdev(retl)}")
    return retl


# %%
parallel = len(ds) < 120
sgdc_accu = mean_sklearn_get_accuracy(ds, SGDClassifier)
rc_accu = mean_sklearn_get_accuracy(ds, RidgeClassifier)
pac_accu = mean_sklearn_get_accuracy(ds, PassiveAggressiveClassifier)
knn_accu = mean_sklearn_get_accuracy(ds, KNeighborsClassifier)
svm_accu = mean_sklearn_get_accuracy(ds, SVC)
dt_accu = mean_sklearn_get_accuracy(
    ds,
    DecisionTreeClassifier,
    parallel=parallel
)
rdn_forest_accu = mean_sklearn_get_accuracy(
    ds,
    RandomForestClassifier,
    parallel=parallel
)
etc_accu = mean_sklearn_get_accuracy(
    ds,
    ExtraTreesClassifier,
    parallel=parallel
)
adaboost_dt_accu = mean_sklearn_get_accuracy(
    ds,
    AdaBoostClassifier,
    model_kwds={"base_estimator": DecisionTreeClassifier()},
    parallel=parallel
)
# gbc_accu = mean_sklearn_get_accuracy(
#         ds,
#         GradientBoostingClassifier,
#         model_kwds={"n_iter_no_change": 5, "tol": 0.01},
#     parallel=parallel
# )
# hgbc_accu = mean_sklearn_get_accuracy(
#     ds,
#     HistGradientBoostingClassifier,
#     model_kwds={"n_iter_no_change": 5, "tol": 0.01},
#     parallel=parallel
#  )
xgb_accu = mean_sklearn_get_accuracy(
    ds,
    XGBClassifier,
    model_kwds={"n_jobs": multiprocessing.cpu_count()}
)
bag_knn_accu = mean_sklearn_get_accuracy(
    ds,
    BaggingClassifier,
    model_kwds={"base_estimator": KNeighborsClassifier()}
)
bag_svm_accu = mean_sklearn_get_accuracy(
    ds,
    BaggingClassifier,
    model_kwds={"base_estimator": SVC()}
)
bag_dt_accu = mean_sklearn_get_accuracy(
    ds,
    BaggingClassifier,
    model_kwds={"base_estimator": DecisionTreeClassifier()}
)
vote_accu = mean_sklearn_get_accuracy(
    ds, VotingClassifier,
    model_kwds={"estimators": [
        ('knn', KNeighborsClassifier()),
        ('svm', SVC()),
        ('dt', DecisionTreeClassifier())
    ]}
)
# mlp_accu = mean_sklearn_get_accuracy(ds, MLPClassifier)

# %%
acu_table = pd.DataFrame({
    "SGD": sgdc_accu,
    "Ridge": rc_accu,
    "Passive-Aggressive": pac_accu,
    "KNN": knn_accu,
    "SVM": svm_accu,
    "Decision Tree": dt_accu,
    "Random Forest": rdn_forest_accu,
    "Extra Trees": etc_accu,
    "AdaBoost (Decision Tree)": adaboost_dt_accu,
    # "GBC": gbc_accu,
    # "HGBC": hgbc_accu,
    "XGBoost": xgb_accu,
    "Bagging (KNN)": bag_knn_accu,
    "Bagging (SVM)": bag_svm_accu,
    "Bagging (Decision Tree)": bag_dt_accu,
    "Voting (KNN, SVM, Decision Tree)": vote_accu
})
p = sns.catplot(data=acu_table, kind="box", height=5, aspect=2, orient="h")
p.set_axis_labels("Classification Algorithm", "Accuracy")
p.set(xlim=(40, 100))
plt.show()

# %%
