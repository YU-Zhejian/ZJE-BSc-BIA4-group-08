from BIA_G8.data_analysis.analysis_config import grid_search

if __name__ == "__main__":
    preprocessor_pipeline_configuration_paths = [
        "pp_plain.toml",
        "pp_adapt_hist.toml",
        "pp_unsharp.toml",
        "pp_scgan.toml"
    ]
    classifier_configuration_paths = [
        "ml_cnn.toml",
        "ml_vote.toml",
        "ml_xgb.toml",
        "ml_svc.toml",
        "ml_extra_trees.toml",
        "ml_rf.toml",
        "ml_resnet50.toml",
    ]
    dataset_configuration_paths = [
        "ds_new.toml",
        "ds_new_nomask.toml",
        "ds_old.toml"
    ]
    grid_search(
        dataset_configuration_paths=dataset_configuration_paths,
        preprocessor_pipeline_configuration_paths=preprocessor_pipeline_configuration_paths,
        classifier_configuration_paths=classifier_configuration_paths,
        out_csv="gs.csv",
        replication=10
    )
