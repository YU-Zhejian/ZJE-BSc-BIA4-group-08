from BIA_G8.data_analysis.covid_dataset_configuration import CovidDatasetConfiguration
from BIA_G8.model.classifier import ToyCNNClassifier, Resnet50Classifier

if __name__ == "__main__":
    width, height = 256, 256
    ds = CovidDatasetConfiguration.load("ds_new.toml").dataset
    ds_train, ds_test = ds.train_test_split()
    ToyCNNClassifier.new(
        hyper_params={
            "batch_size": 16,
            "num_epochs": 20,
            "lr": 0.00001,
            "device": "cuda"
        },
        model_params={
            "n_features": width * height,
            "n_classes": ds.n_classes,
            "kernel_size": 3,
            "stride": 2,
            "padding": 1
        }
    ).diagnostic_fit(
        train_dataset=ds_train,
        test_dataset=ds_test,
        output_diagnoistics_path="diagnose_cnn.csv",
        num_epochs=200
    )
    Resnet50Classifier.new(
        hyper_params={
            "batch_size": 16,
            "num_epochs": 20,
            "lr": 0.00001,
            "device": "cuda"
        },
        model_params={
            "n_features": width * height,
            "n_classes": ds.n_classes,
            "kernel_size": 3,
            "stride": 2,
            "padding": 1
        }
    ).diagnostic_fit(
        train_dataset=ds_train,
        test_dataset=ds_test,
        output_diagnoistics_path="diagnose_resnet50.csv",
        num_epochs=200
    )
