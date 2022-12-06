from BIA_G8.data_analysis.brute_force_grid_search import grid_search

if __name__ == "__main__":
    preprocessor_pipeline_configuration_paths = [
        "/home/yuzj/Documents/2022-23-Group-08/src/BIA_G8_DATA_ANALYSIS/pp_plain.toml",
        "/home/yuzj/Documents/2022-23-Group-08/src/BIA_G8_DATA_ANALYSIS/pp_adapt_hist.toml",
        "/home/yuzj/Documents/2022-23-Group-08/src/BIA_G8_DATA_ANALYSIS/pp_unsharp.toml",
    ]
    classifier_configuration_paths = [
        "/home/yuzj/Documents/2022-23-Group-08/src/BIA_G8_DATA_ANALYSIS/cnn.toml",
        "/home/yuzj/Documents/2022-23-Group-08/src/BIA_G8_DATA_ANALYSIS/vote.toml",
        "/home/yuzj/Documents/2022-23-Group-08/src/BIA_G8_DATA_ANALYSIS/xgb.toml",
        "/home/yuzj/Documents/2022-23-Group-08/src/BIA_G8_DATA_ANALYSIS/knn.toml",
        "/home/yuzj/Documents/2022-23-Group-08/src/BIA_G8_DATA_ANALYSIS/svc.toml",
        "/home/yuzj/Documents/2022-23-Group-08/src/BIA_G8_DATA_ANALYSIS/extra_trees.toml",
        "/home/yuzj/Documents/2022-23-Group-08/src/BIA_G8_DATA_ANALYSIS/rf.toml",
        "/home/yuzj/Documents/2022-23-Group-08/src/BIA_G8_DATA_ANALYSIS/resnet50.toml"
    ]
    grid_search(
        dataset_path="/home/yuzj/Documents/2022-23-Group-08/data_analysis/covid_image_new_nomask",
        encoder_dict={
            "COVID": 0,
            "Lung_Opacity": 1,
            "Normal": 2,
            "Viral Pneumonia": 3,
        },
        preprocessor_pipeline_configuration_paths=preprocessor_pipeline_configuration_paths,
        classifier_configuration_paths=classifier_configuration_paths,
        n_data_to_load=600,
        n_classes=4,
        out_csv="gs_new_data_nomask.csv",
        replication=10
    )
    grid_search(
        dataset_path="/home/yuzj/Documents/2022-23-Group-08/data_analysis/covid_image",
        encoder_dict={
            "COVID-19": 0,
            "NORMAL": 1,
            "Viral_Pneumonia": 2
        },
        preprocessor_pipeline_configuration_paths=preprocessor_pipeline_configuration_paths,
        classifier_configuration_paths=classifier_configuration_paths,
        n_data_to_load=600,
        n_classes=3,
        out_csv="gs_old_data.csv",
        replication=10
    )
    grid_search(
        dataset_path="/home/yuzj/Documents/2022-23-Group-08/data_analysis/covid_image_new",
        encoder_dict={
            "COVID": 0,
            "Lung_Opacity": 1,
            "Normal": 2,
            "Viral Pneumonia": 3,
        },
        preprocessor_pipeline_configuration_paths=preprocessor_pipeline_configuration_paths,
        classifier_configuration_paths=classifier_configuration_paths,
        n_data_to_load=600,
        n_classes=4,
        out_csv="gs_new_data.csv",
        replication=10
    )
