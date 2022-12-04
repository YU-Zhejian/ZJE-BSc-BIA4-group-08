from __future__ import annotations

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from BIA_G8.model import BaseSklearnClassifier


class SklearnKNearestNeighborsClassifier(BaseSklearnClassifier):
    _name = "SklearnKNearestNeighborsClassifier"
    _model_type = KNeighborsClassifier


if __name__ == '__main__':
    x, y = make_classification(
        n_samples=12000,
        n_features=7,
        n_classes=3,
        n_clusters_per_class=1
    )
    x_train, x_test, y_train, y_test = train_test_split(x, y)

    m = SklearnKNearestNeighborsClassifier.new()
    m.fit(x_train, y_train)
    m.save("tmp.pkl.xz")
    del m
    m2 = SklearnKNearestNeighborsClassifier.load("tmp.pkl.xz")
    print(np.sum(m2.predict(x_test) == y_test) * 100 / len(y_test))
