library(tidyverse)

data <- readr::read_csv("diagnose_cnn.csv")
ggplot(data) +
    geom_line(aes(x = epoch, y = train_loss), color = "red") +
    geom_line(aes(x = epoch, y = test_loss), color = "blue") +
    theme_bw() +
    ylim(0, NA)
ggplot(data) +
    geom_line(aes(x = epoch, y = train_accu), color = "red") +
    geom_line(aes(x = epoch, y = test_accu), color = "blue") +
    theme_bw() +
    ylim(0, 1)

data <- readr::read_csv("diagnose_resnet50.csv")
ggplot(data) +
    geom_line(aes(x = epoch, y = train_loss), color = "red") +
    geom_line(aes(x = epoch, y = test_loss), color = "blue") +
    theme_bw() +
    ylim(0, NA)
ggplot(data) +
    geom_line(aes(x = epoch, y = train_accu), color = "red") +
    geom_line(aes(x = epoch, y = test_accu), color = "blue") +
    theme_bw() +
    ylim(0, 1)

