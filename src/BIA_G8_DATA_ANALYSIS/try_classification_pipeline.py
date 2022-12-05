from typing import Type

from BIA_G8.model.classifier import load_classifier, ToyCNNClassifier, \
    AbstractClassifier, XGBoostClassifier, SklearnVotingClassifier
from BIA_G8_DATA_ANALYSIS.covid_dataset import generate_fake_classification_dataset, CovidDataSet


def run(
        _ds_train: CovidDataSet,
        _ds_test: CovidDataSet,
        save_path: str,
        classifier_type: Type[AbstractClassifier],
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
    run(ds_train, ds_test, "xgb.toml", XGBoostClassifier)
    run(ds_train, ds_test, "vote.toml", SklearnVotingClassifier)
    run(
        ds_train, ds_test, "cnn.toml", ToyCNNClassifier,
        hyper_params={
            "batch_size": 17,
            "num_epochs": 5,
            "lr": 0.0001
        },
        model_params={
            "n_features": width * height,
            "n_classes": ds.n_classes,
            "kernel_size": 3,
            "stride": 2,
            "padding": 1
        }
    )
