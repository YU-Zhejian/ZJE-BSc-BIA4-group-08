from BIA_G8.data_analysis.analysis_config import AnalysisConfiguration

if __name__ == "__main__":
    ac = AnalysisConfiguration(
        dataset_path="/home/yuzj/Documents/2022-23-Group-08/data_analysis/covid_image_new",
        encoder_dict={
            "COVID": 0,
            "Lung_Opacity": 1,
            "Normal": 2,
            "Viral Pneumonia": 3
        },
        preprocessor_pipeline_configuration_path="pp_adapt_hist.toml",
        classifier_configuration_path="cnn.toml",
        n_data_to_load=600,
        n_classes=4
    )
    ac.pre_process()
    print(ac.ml())
    ac.save("ac.toml")

    ac2 = ac.load("ac.toml")
    ac2.pre_process().ml()
