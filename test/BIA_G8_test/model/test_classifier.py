import glob
import os
import tempfile
from typing import Type

import numpy as np

from BIA_G8.data_analysis.covid_dataset import generate_fake_classification_dataset
from BIA_G8.model.classifier import load_classifier, ToyCNNClassifier, \
    ClassifierInterface, XGBoostClassifier, SklearnVotingClassifier, SklearnSupportingVectorMachineClassifier, \
    SklearnExtraTreesClassifier, SklearnRandomForestClassifier, SklearnKNearestNeighborsClassifier

width, height = 16, 16
ds = generate_fake_classification_dataset(
    size=12,
    width=width,
    height=height
)
ds_train, ds_test = ds.train_test_split()


def run(
        classifier_type: Type[ClassifierInterface],
        **kwargs
):
    global ds_train, ds_test, width, height
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "tmp.toml")
        classifier_type.new(**kwargs).fit(ds_train).save(save_path)
        m2 = load_classifier(save_path)
        _ = m2.predict(np.zeros(shape=(width, height), dtype=int))
        _ = m2.predicts([np.zeros(shape=(width, height), dtype=int)] * 1024)
        _ = m2.evaluate(ds_test)
        _ = map(os.remove, glob.glob(save_path + "*"))


# Disabled, too slow.
# def test_resnet50():
#     run(
#         ds_train,
#         ds_test,
#         "ml_resnet50.toml",
#         Resnet50Classifier,
#         hyper_params={
#             "batch_size": 4,
#             "num_epochs": 5,
#             "lr": 0.01,
#             "device": "cuda"
#         },
#         model_params={
#             "block": 1,
#             "layers": 1
#         }
#     )


def test_xgb():
    run(XGBoostClassifier, tree_method="gpu_hist")


def test_svc():
    run(SklearnSupportingVectorMachineClassifier)


def test_extra_trees():
    run(SklearnExtraTreesClassifier)


def test_rf():
    run(SklearnRandomForestClassifier)


def test_knn():
    run(SklearnKNearestNeighborsClassifier)


def test_vote():
    run(SklearnVotingClassifier)


def test_cnn():
    run(ToyCNNClassifier, hyper_params={
        "batch_size": 4,
        "num_epochs": 5,
        "lr": 0.01,
        "device": "cuda"
    }, model_params={
        "n_features": width * height,
        "n_classes": ds.n_classes,
        "kernel_size": 3,
        "stride": 2,
        "padding": 1
    })
