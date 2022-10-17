import joblib
import numpy as np
import numpy.typing as npt
import ray
from ray.util.joblib import register_ray
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import confusion_matrix

from BIA_COVID_CLASS.covid_helper import covid_dataset
from BIA_G8 import get_lh

_lh = get_lh(__name__)

if __name__ == '__main__':
    if not ray.is_initialized():
        ray.init()
    register_ray()
    with joblib.parallel_backend('ray'):
        df = covid_dataset.get_ray_dataset(
            "/media/yuzj/BUP/covid19-database-np",
            size=120
        )
