import glob
import os
from typing import Type

from BIA_G8.data_analysis.covid_dataset import generate_fake_classification_dataset, CovidDataSet
from BIA_G8.model.classifier import load_classifier, ToyCNNClassifier, \
    ClassifierInterface, XGBoostClassifier, SklearnVotingClassifier, SklearnSupportingVectorMachineClassifier, \
    SklearnExtraTreesClassifier, SklearnRandomForestClassifier, SklearnKNearestNeighborsClassifier, Resnet50Classifier


def run(
        _ds_train: CovidDataSet,
        _ds_test: CovidDataSet,
        save_path: str,
        classifier_type: Type[ClassifierInterface],
        **kwargs
):
    classifier_type.new(**kwargs).fit(_ds_train).save(save_path)
    m2 = load_classifier(save_path)
    _ = m2.evaluate(_ds_test)
    _ = map(os.remove, glob.glob(save_path + "*"))


width, height = 16, 16
ds = generate_fake_classification_dataset(
    size=12,
    width=width,
    height=height
)
ds_train, ds_test = ds.train_test_split()


def test_resnet50():
    run(
        ds_train,
        ds_test,
        "resnet50.toml",
        Resnet50Classifier,
        hyper_params={
            "batch_size": 4,
            "num_epochs": 5,
            "lr": 0.01,
            "device": "cuda"
        },
        model_params={
            "block": 1,
            "layers": 1
        }
    )


def test_xgb():
    run(
        ds_train,
        ds_test,
        "xgb.toml",
        XGBoostClassifier,
        tree_method="gpu_hist"
    )


def test_svc():
    run(
        ds_train,
        ds_test,
        "svc.toml",
        SklearnSupportingVectorMachineClassifier
    )


def test_extra_trees():
    run(
        ds_train,
        ds_test,
        "extra_trees.toml",
        SklearnExtraTreesClassifier
    )


def test_rf():
    run(
        ds_train,
        ds_test,
        "rf.toml",
        SklearnRandomForestClassifier
    )


def test_knn():
    run(
        ds_train,
        ds_test,
        "knn.toml",
        SklearnKNearestNeighborsClassifier
    )


def test_vote():
    run(
        ds_train,
        ds_test,
        "vote.toml",
        SklearnVotingClassifier
    )


def test_cnn():
    run(
        ds_train, ds_test, "cnn.toml", ToyCNNClassifier,
        hyper_params={
            "batch_size": 4,
            "num_epochs": 5,
            "lr": 0.01,
            "device": "cuda"
        },
        model_params={
            "n_features": width * height,
            "n_classes": ds.n_classes,
            "kernel_size": 3,
            "stride": 2,
            "padding": 1
        }
    )
