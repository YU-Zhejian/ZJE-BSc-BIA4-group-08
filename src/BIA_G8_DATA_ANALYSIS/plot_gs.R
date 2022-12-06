library(tidyverse)

gs_new_data <- readr::read_csv("gs_new_data.csv") %>%
    dplyr::mutate(
        data="new"
    )

gs_new_data_nomask <- readr::read_csv("gs_new_data_nomask.csv") %>%
    dplyr::mutate(
        data="new_nomask"
    )

gs_old_data <- readr::read_csv("gs_old_data.csv") %>%
    dplyr::mutate(
        data="old"
    )

gs <- dplyr::bind_rows(gs_new_data, gs_new_data_nomask, gs_old_data)
ggplot(gs) +
    geom_boxplot(aes(
        y=classifier_configuration_path,
        x=accuracy,
        fill=classifier_configuration_path
    )) +
    facet_wrap(preprocessor_pipeline_configuration_path~data) +
    theme_bw()

