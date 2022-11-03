import numpy as np
import numpy.typing as npt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from BIA_G8 import get_lh
from BIA_G8.covid_helper import covid_dataset

_lh = get_lh(__name__)


def train_model(
        np_image_1d: npt.NDArray,
        label_int: npt.NDArray
) -> SVC:
    _lh.info("Training started with %d cases", len(np_image_1d))
    svc = SVC()
    svc.fit(X=np_image_1d, y=label_int)
    _lh.info("Training finished")
    return svc


def predict_model(
        model: SVC,
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
    _lh.info("Evaluation finished with accuracy=%.2f%% cmatrix=\n%s", accuracy * 100, str(_confusion_matrix))
    return accuracy


if __name__ == '__main__':
    X, y = covid_dataset.CovidDataSet.from_directory(
        "/media/yuzj/BUP/covid19-database-np",
        size=600
    ).sklearn_dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True)
    model = train_model(X_train, y_train)
    evaluate(model.predict(X_test), y_test)
