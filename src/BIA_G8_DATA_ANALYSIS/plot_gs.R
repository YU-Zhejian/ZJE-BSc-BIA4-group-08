library(tidyverse)

gs <- readr::read_csv("gs.csv") %>%
    dplyr::filter(
        classifier_configuration_path!="ml_knn.toml"
    )
ggplot(gs) +
    geom_boxplot(aes(
        y=classifier_configuration_path,
        x=accuracy,
        fill=classifier_configuration_path
    )) +
    facet_grid(
        preprocessor_pipeline_configuration_path~dataset_configuration_path,
        scales="free"
    ) +
    theme_bw()

