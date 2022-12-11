import sys
import webbrowser
from typing import Optional, Dict

import numpy.typing as npt
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QMessageBox, QTableWidgetItem

from BIA_G8 import get_lh
from BIA_G8._main import perform_ml
from BIA_G8._ui.classification import Ui_ClassificationMainWindow
from BIA_G8.model.classifier import ClassifierInterface, load_classifier
from BIA_G8.model.preprocesor_pipeline import PreprocessorPipeline

_lh = get_lh(__name__)


class ClassificationWindow(QMainWindow):
    _loaded_data: Optional[Dict[str, npt.NDArray]]
    _predictions: Optional[Dict[str, str]]
    _loaded_classifier: Optional[ClassifierInterface]
    _loaded_pp: Optional[PreprocessorPipeline]

    def __init__(self):
        self._loaded_data = None
        self._loaded_classifier = None
        self._loaded_pp = None
        super().__init__()
        _lh.info("Classification Main Window Initializing...")

        self.ui = Ui_ClassificationMainWindow()
        self.ui.setupUi(self)
        self.ui.actionClose.triggered.connect(lambda: self.close())
        self.ui.actionOfficialSite.triggered.connect(
            lambda: webbrowser.open_new_tab("https://gitee.com/yuzjlab/2022-23-group-08")
        )
        self.ui.actionOpen_Image_Folder.triggered.connect(
            lambda: self.load_data()
        )
        self.ui.actionOpen_Preprocessor_Pipeline_Config.triggered.connect(
            lambda: self.load_pp()
        )
        self.ui.actionOpen_Machine_Learning_Model.triggered.connect(
            lambda: self.load_classifier()
        )
        self.ui.actionExecute.triggered.connect(
            lambda: self.predict()
        )
        self.ui.actionClear.triggered.connect(
            lambda: self.clear()
        )
        self.ui.actionSave_Predicted_Outcome.triggered.connect(
            lambda: self.save_predicted()
        )
        self.clear()

    def load_data(self):
        self._loaded_data = None
        data_path = QFileDialog.getExistingDirectory(
            self,
            caption='Select directory containing images',
            directory="."
        )
        if data_path == "":
            QMessageBox.warning(self, 'WARNING', "You did not select any directory.")
            return
        try:
            self._loaded_data = perform_ml.load_data(data_path)
        except Exception as e:
            QMessageBox.critical(self, 'ERROR', f"Exception {e} captured!")
            self._loaded_data = None
            return
        if not self._loaded_data:
            QMessageBox.critical(self, 'ERROR', "No data loaded!")
            self._loaded_data = None
            return
        self.ui.actionOpen_Image_Folder.setText(f"Open Image Folder -- {data_path}")

    def load_classifier(self):
        self._loaded_classifier = None
        filename, _ = QFileDialog.getOpenFileName(
            self,
            caption='Select File',
            directory=".",
            filter='TOML Classifier Config (*.toml);;ALL (*.*)'
        )
        if filename == "":
            QMessageBox.warning(self, 'WARNING', "You did not select any file.")
            return
        try:
            self._loaded_classifier = load_classifier(filename)
        except Exception as e:
            QMessageBox.critical(self, 'ERROR', f"Exception {e} captured!")
            self._loaded_classifier = None
            return
        self.ui.actionOpen_Machine_Learning_Model.setText(f"Open Machine Learning Model -- {filename}")

    def load_pp(self):
        self._loaded_pp = None
        filename, _ = QFileDialog.getOpenFileName(
            self,
            caption='Select File',
            directory=".",
            filter='TOML Preprocessor Pipeline Config (*.toml);;ALL (*.*)'
        )
        if filename == "":
            QMessageBox.warning(self, 'WARNING', "You did not select any file.")
            return
        try:
            self._loaded_pp = PreprocessorPipeline.load(filename)
        except Exception as e:
            QMessageBox.critical(self, 'ERROR', f"Exception {e} captured!")
            self._loaded_pp = None
            return
        self.ui.actionOpen_Preprocessor_Pipeline_Config.setText(f"Open Preprocessor Pipeline Config -- {filename}")

    def clear(self):
        self.ui.tableWidget.clear()
        self._loaded_data = None
        self._loaded_classifier = None
        self._loaded_pp = None
        self._predictions = None
        self.ui.actionOpen_Machine_Learning_Model.setText("Open Machine Learning Model")
        self.ui.actionOpen_Image_Folder.setText("Open Image Folder")
        self.ui.actionOpen_Preprocessor_Pipeline_Config.setText("Open Preprocessor Pipeline Config")

    def save_predicted(self):
        if self._predictions is None:
            QMessageBox.warning(self, 'WARNING', "Prediction had not been initiated done")
            return
        filename, _ = QFileDialog.getSaveFileName(
            self,
            caption='Select File',
            directory=".",
            filter='Common Separated Value (*.csv);;ALL (*.*)'
        )
        if filename == "":
            QMessageBox.warning(self, 'WARNING', "You did not select any file.")
            return
        perform_ml.save(self._predictions, filename)

    def predict(self):
        self.statusBar().showMessage("Start prediction...")
        if any((
                self._loaded_pp is None,
                self._loaded_data is None,
                self._loaded_classifier is None
        )):
            QMessageBox.warning(
                self,
                'WARNING',
                "Pre-processing Pipeline, data or classifier is not loaded."
            )
            return

        try:
            self._predictions = perform_ml.predict(
                self._loaded_data,
                self._loaded_pp,
                self._loaded_classifier
            )
        except Exception as e:
            QMessageBox.critical(self, 'ERROR', f"Exception {e} captured!")
            self._predictions = None
            return
        self.ui.tableWidget.setRowCount(len(self._predictions))
        self.ui.tableWidget.setColumnCount(2)
        self.statusBar().showMessage("Updating table view...")
        for i, (file_path, predict) in enumerate(self._predictions.items()):
            self.ui.tableWidget.setItem(i, 0, QTableWidgetItem(file_path))
            self.ui.tableWidget.setItem(i, 1, QTableWidgetItem(predict))

        self.statusBar().showMessage("Prediction Finished!")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = ClassificationWindow()
    ui.show()
    sys.exit(app.exec_())
