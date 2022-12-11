import sys
from typing import Optional

import seaborn as sns
import skimage.transform as skitrans
from PyQt5.QtWidgets import QFileDialog, QMainWindow, QApplication
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

from BIA_G8._ui.data_explorer import Ui_DataExplorerMainWIndow
from BIA_G8.data_analysis.covid_dataset import CovidDataSet


class DataExplorer(QMainWindow):
    _dataset_path: Optional[str]

    def __init__(self):
        super().__init__()
        self.ui = Ui_DataExplorerMainWIndow()
        self.ui.setupUi(self)
        self.ui.pushButton.clicked.connect(self.upload)
        self.ui.pushButton_2.clicked.connect(self.start)
        self._dataset_path = None

    def upload(self):
        self._dataset_path = QFileDialog.getExistingDirectory(
            self,
            caption='Select COVID Dataset Directory',
            directory='.'
        )
        self.ui.lineEdit.setText(self._dataset_path)

    def start(self):
        ds = CovidDataSet.parallel_from_directory(
            self._dataset_path
        ).parallel_apply(
            lambda img: img[:, :, 0] if len(img.shape) == 3 else img
        ).parallel_apply(
            lambda img: skitrans.resize(img, (256, 256))
        )
        ds_tsne_transformed = TSNE(
            learning_rate=200,
            n_iter=1000,
            init="random"
        ).fit_transform(ds.sklearn_dataset[0])
        sns.scatterplot(
            x=ds_tsne_transformed[:, 0],
            y=ds_tsne_transformed[:, 1],
            hue=list(map(ds.decode, ds.sklearn_dataset[1]))
        )
        plt.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = DataExplorer()
    ui.show()
    sys.exit(app.exec_())
