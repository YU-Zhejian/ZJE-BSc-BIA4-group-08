import os
import sys
from typing import Optional

import numpy.typing as npt
import skimage.io as skiio
from PyQt5.QtWidgets import QFileDialog, QApplication, QMainWindow, QMessageBox

from BIA_G8 import get_lh
from BIA_G8._main._method_selector_gui import PreprocessorPipelineWindow
from BIA_G8._ui.preprocessor_explorer_main_page import Ui_PreprocessorExplorerMainWindow

_lh = get_lh(__name__)


class HomePage(QMainWindow):
    _orig_image: Optional[npt.NDArray]

    def __init__(self):
        self._orig_image = None
        super().__init__()
        _lh.info("HomePage Initializing...")
        self.ui = Ui_PreprocessorExplorerMainWindow()
        self.ui.setupUi(self)
        self.ui.pushButton_3.clicked.connect(self.upload)
        self.ui.pushButton.clicked.connect(self.open_model_selector)
        self.ui.pushButton_4.clicked.connect(lambda: self.close())
        self.ui.pushButton_3.clicked.connect(
            lambda: self.ui.label_3.setText('Click "Start" to select the preprocessing method.')
        )
        self.ui.pushButton_4.clicked.connect(
            lambda: self.ui.label_5.setText('Thanks for your using!')
        )
        _lh.info("HomePage Initialized")

    def upload(self) -> None:
        _lh.info("Homepage: Waiting for Filename")
        picture_dir = os.path.join(os.path.expanduser("~"), "Pictures")
        if not os.path.exists(picture_dir):
            picture_dir = "."

        filename, _ = QFileDialog.getOpenFileName(
            self,
            caption='Select File',
            directory=picture_dir,
            filter='SKImage-Parsable Image Files (*.png *.jpg *.jpeg *.tif *.tiff);;All (*.*)'
        )
        if filename == "":
            QMessageBox.warning(self, 'WARNING', "You did not select any image.")
            return
        _lh.info("Homepage: get filename '%s'", filename)
        try:
            self._orig_image = skiio.imread(filename)
        except Exception as e:
            QMessageBox.critical(self, 'ERROR', f"Exception {e} captured!")
            exit(1)
        _lh.info("Homepage: read filename '%s' success", filename)
        self.ui.label_4.setText('Click "Output" to save your configuration.')

    def open_model_selector(self):
        if self._orig_image is None:
            QMessageBox.warning(self, 'WARNING', "You did not select any image.")
            return
        PreprocessorPipelineWindow(self._orig_image).exec()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = HomePage()
    ui.show()
    sys.exit(app.exec_())
