import joblib
import numpy as np
import numpy.typing as npt
import ray
from ray.util.joblib import register_ray
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as KNN

from BIA_G8.covid_helper import covid_dataset
from BIA_G8 import get_lh
from BIA_G8.helper import ml_helper

_lh = get_lh(__name__)


def train_model(
        np_image_1d: npt.NDArray,
        label_int: npt.NDArray
) -> KNN:
    _lh.info("Training started with %d cases", len(np_image_1d))
    knn = KNN()
    knn.fit(X=np_image_1d, y=label_int)
    _lh.info("Training finished")
    return knn


def predict_model(
        model: KNN,
        np_image_1d: npt.NDArray,
) -> npt.NDArray:
    _lh.info("Prediction started with %d cases", len(np_image_1d))
    result = model.predict(X=np_image_1d)
    _lh.info("Prediction finished")
    return result


def evaluate(result_1: npt.NDArray, result_2: npt.NDArray) -> float:
    _lh.info("Evaluation started with %d cases", len(result_2))
    accuracy = np.sum(result_1 == result_2) / len(result_2)
    _confusion_matrix = confusion_matrix(result_1, result_2)
    _lh.info(
        "Evaluation finished with accuracy=%.2f%% cmatrix=\n%s",
        accuracy * 100,
        ml_helper.print_confusion_matrix(_confusion_matrix, ["COVID-19", "NORMAL",  "Viral_Pneumonia"])
    )
    return accuracy


if __name__ == '__main__':
    if not ray.is_initialized():
        ray.init()
    register_ray()
    dataset = covid_dataset.CovidDataSet.parallel_from_directory(
        "/media/yuzj/BUP/covid19-database-np",
        size=600
    )
    with joblib.parallel_backend('ray'):
        X, y = dataset.get_sklearn_dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True)
        print("Train-test-Split FIN")
        model = train_model(X_train, y_train)
        evaluate(model.predict(X_test), y_test)
