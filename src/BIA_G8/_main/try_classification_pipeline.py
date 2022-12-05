import skimage.transform as skitrans

from BIA_G8.model.classifier import XGBoostClassifier, load_classifier, SklearnVotingClassifier, ToyCNNClassifier
from BIA_G8_DATA_ANALYSIS.covid_dataset import generate_fake_classification_dataset

if __name__ == '__main__':
    ds = generate_fake_classification_dataset(120).parallel_apply(
        lambda img: skitrans.resize(
            img,
            (256, 256)
        )
    )
    ds_train, ds_test = ds.train_test_split()

    XGBoostClassifier.new().fit(ds_train).save("xgb.toml")
    m2 = load_classifier("xgb.toml")
    print(m2.evaluate(ds_test))

    SklearnVotingClassifier.new().fit(ds_train).save("vote.toml")
    m2 = load_classifier("vote.toml")
    print(m2.evaluate(ds_test))

    ToyCNNClassifier.new(
        hyper_params={
            "batch_size": 17,
            "num_epochs": 5,
            "lr": 0.0001
        },
        model_params={
            "n_features": 128 * 128,
            "n_classes": 3,
            "kernel_size": 3,
            "stride": 2,
            "padding": 1
        }
    ).fit(ds_train).save("cnn.toml")
    m2 = load_classifier("cnn.toml")
    print(m2.evaluate(ds_test))
