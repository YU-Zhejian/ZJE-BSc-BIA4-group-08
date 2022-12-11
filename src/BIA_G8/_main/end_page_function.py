from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication,QMainWindow
import sys

import click

from BIA_G8._ui.attention_page import Ui_Form


class accept(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)



if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = accept()
    ui.show()
    sys.exit(app.exec_())
