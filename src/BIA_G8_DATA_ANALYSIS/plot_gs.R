library(tidyverse)

gs <- readr::read_csv("gs.csv") %>%
    dplyr::filter(
        classifier_configuration_path != "ml_knn.toml"
    )
g <- ggplot(gs) +
    geom_violin(aes(
        y = classifier_configuration_path,
        x = accuracy,
        fill = classifier_configuration_path
    )) +
    facet_grid(
        preprocessor_pipeline_configuration_path ~ dataset_configuration_path,
        scales = "free"
    ) +
    theme_bw()

ggsave(
    "gs.png",
    plot=g,
    width=10,
    height=5
)

g <- ggplot(gs) +
    geom_violin(aes(
        y = preprocessor_pipeline_configuration_path,
        x = accuracy,
        fill = preprocessor_pipeline_configuration_path
    )) +
    facet_grid(
        . ~ dataset_configuration_path,
        scales = "free"
    ) +
    theme_bw()

ggsave(
    "gs_pp.png",
    plot=g,
    width=10,
    height=5
)

g <- ggplot(gs) +
    geom_violin(aes(
        y = classifier_configuration_path,
        x = accuracy,
        fill = classifier_configuration_path
    )) +
    facet_grid(
        . ~ dataset_configuration_path,
        scales = "free"
    ) +
    theme_bw()

ggsave(
    "gs_classifier.png",
    plot=g,
    width=10,
    height=5
)
g <- ggplot(gs) +
    geom_violin(aes(
        y = dataset_configuration_path,
        x = accuracy,
        fill = dataset_configuration_path
    )) +
    theme_bw()

ggsave(
    "gs_dataset.png",
    plot=g,
    width=10,
    height=5
)
