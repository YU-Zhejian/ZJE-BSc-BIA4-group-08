import sys

import numpy.typing as npt
import skimage.io as skiio
from PyQt5.QtWidgets import QFileDialog, QApplication, QMainWindow, QMessageBox

from BIA_G8 import get_lh
from BIA_G8._main import method_page_function
from BIA_G8._ui.main_page import Ui_MainWindow

_lh = get_lh(__name__)


class HomePage(QMainWindow):
    _orig_image: npt.NDArray

    def __init__(self):
        super().__init__()
        _lh.info("HomePage Initializing...")
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.pushButton_3.clicked.connect(self.upload)
        self.ui.pushButton.clicked.connect(self.openwindow1)
        self.ui.pushButton_4.clicked.connect(lambda: self.close())
        self.ui.pushButton_3.clicked.connect(
            lambda: self.ui.label_3.setText('Click "Start" to select the preprocessing method.')
        )
        self.ui.pushButton.clicked.connect(
            lambda: self.ui.label_4.setText('Click "Output" to save your configuration.'))
        self.ui.pushButton_4.clicked.connect(
            lambda: self.ui.label_5.setText('Thanks for your using!')
        )
        _lh.info("HomePage Initialized")

    def upload(self):
        _lh.info("Homepage: Waiting for Filename")
        filename, _ = QFileDialog.getOpenFileName(
            self,
            caption='Select File',
            directory='',
            filter='SKImage-Parsable Image Files (*.png *.jpg *.jpeg *.tif *.tiff);;All (*.*)'
        )  # FIXME: Need one image only
        _lh.info("Homepage: get filename %s", filename)
        try:
            self._orig_image = skiio.imread(filename)
        except Exception as e:
            QMessageBox.critical(self, title='ERROR', text=f"Exception {e} captured!")
            exit(1)
        _lh.info("Homepage: read filename '%s' success", filename)

    def openwindow1(self):
        self.m = method_page_function.link_method(self._ori_img)
        self.m.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = HomePage()
    ui.show()
    sys.exit(app.exec_())
