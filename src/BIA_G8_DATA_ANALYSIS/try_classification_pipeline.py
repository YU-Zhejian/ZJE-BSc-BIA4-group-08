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
    print(m2.evaluate(_ds_test))


if __name__ == '__main__':
    width, height = 256, 256
    ds = generate_fake_classification_dataset(
        size=120,
        width=width,
        height=height
    )
    ds_train, ds_test = ds.train_test_split()
    run(
        ds_train,
        ds_test,
        "ml_resnet50.toml",
        Resnet50Classifier,
        hyper_params={
            "batch_size": 17,
            "num_epochs": 20,
            "lr": 0.0001,
            "device": "cuda"
        },
        model_params={
            "block": 1,
            "layers": 1
        }
    )
    run(
        ds_train,
        ds_test,
        "ml_xgb.toml",
        XGBoostClassifier,
        tree_method="gpu_hist"
    )
    run(
        ds_train,
        ds_test,
        "ml_svc.toml",
        SklearnSupportingVectorMachineClassifier
    )
    run(
        ds_train,
        ds_test,
        "ml_extra_trees.toml",
        SklearnExtraTreesClassifier
    )
    run(
        ds_train,
        ds_test,
        "ml_rf.toml",
        SklearnRandomForestClassifier
    )
    run(
        ds_train,
        ds_test,
        "ml_knn.toml",
        SklearnKNearestNeighborsClassifier
    )
    run(
        ds_train,
        ds_test,
        "ml_vote.toml",
        SklearnVotingClassifier
    )
    run(
        ds_train, ds_test, "ml_cnn.toml", ToyCNNClassifier,
        hyper_params={
            "batch_size": 17,
            "num_epochs": 20,
            "lr": 0.0001,
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
