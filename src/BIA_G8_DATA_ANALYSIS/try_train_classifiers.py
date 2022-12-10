from typing import Type

import skimage.transform as skitrans

from BIA_G8.data_analysis.covid_dataset import CovidDataSet
from BIA_G8.data_analysis.covid_dataset_configuration import CovidDatasetConfiguration
from BIA_G8.helper.ndarray_helper import scale_np_array
from BIA_G8.model.classifier import ToyCNNClassifier, \
    ClassifierInterface, XGBoostClassifier, SklearnVotingClassifier, SklearnSupportingVectorMachineClassifier, \
    SklearnExtraTreesClassifier, SklearnRandomForestClassifier, SklearnKNearestNeighborsClassifier, Resnet50Classifier
from BIA_G8.model.upscaller_preprocessor import SCGANUpscaler


def run(
        _ds: CovidDataSet,
        save_path: str,
        classifier_type: Type[ClassifierInterface],
        **kwargs
):
    classifier_type.new(**kwargs).fit(_ds.sample(10)).save(save_path)


if __name__ == '__main__':
    width, height = 256, 256
    ds = CovidDatasetConfiguration.load("ds_new_nomask_full.toml").dataset.parallel_apply(
        lambda img: img[:, :, 0] if len(img.shape) == 3 else img
    ).parallel_apply(lambda img: skitrans.resize(img,(width, height))
    ).parallel_apply(
        scale_np_array
    )


    run(ds, "ml_resnet50.toml", Resnet50Classifier, hyper_params={
        "batch_size": 17,
        "num_epochs": 50,
        "lr": 0.0001,
        "device": "cuda"
    }, model_params={
        "block": 1,
        "layers": 1
    })

    run(ds, "ml_xgb.toml", XGBoostClassifier)
    run(ds, "ml_svc.toml", SklearnSupportingVectorMachineClassifier)
    run(ds, "ml_extra_trees.toml", SklearnExtraTreesClassifier)
    run(ds, "ml_rf.toml", SklearnRandomForestClassifier)
    run(ds, "ml_knn.toml", SklearnKNearestNeighborsClassifier)
    run(ds, "ml_vote.toml", SklearnVotingClassifier)
    run(ds, "ml_cnn.toml", ToyCNNClassifier, hyper_params={
        "batch_size": 17,
        "num_epochs": 50,
        "lr": 0.0001,
        "device": "cuda"
    }, model_params={
        "n_features": width * height,
        "n_classes": ds.n_classes,
        "kernel_size": 3,
        "stride": 2,
        "padding": 1
    })
    SCGANUpscaler.new(
        generator_params={
            "large_kernel_size": 9,
            "small_kernel_size": 3,
            "n_intermediate_channels": 64,
            "n_blocks": 16,
            "scale_factor": 2,
            "in_channels": 3
        },
        discriminator_params={
            "kernel_size": 3,
            "n_channels": 64,
            "n_blocks": 8,
            "fc_size": 1024,
            "in_channels": 3
        },
        truncated_vgg19_params={
            "i": 5,
            "j": 4
        },
        hyper_params={
            "num_epochs": 130,
            "lr": 0.0001,
            "device": "cuda",
            "batch_size": 16,
            "beta": 0.001,
            "scale_factor": 2
        }
    ).fit(
        dataset=ds.parallel_apply(
            lambda img: skitrans.resize(
                img,
                (128, 128)
            ),
            desc="Scaling to wanted size..."
        )
    ).save("scgan.toml")
