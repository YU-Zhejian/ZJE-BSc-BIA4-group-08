from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy.typing as npt
from PyQt5.QtWidgets import QDialog, QPushButton, QVBoxLayout
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from BIA_G8._ui.method_selector import Ui_MethodSelector
from BIA_G8.model.preprocesor_pipeline import PreprocessorPipeline
from BIA_G8.model.preprocessor import get_preprocessor, AbstractPreprocessor

matplotlib.use("agg")


class ImageDisplayDialog(QDialog):
    _img: npt.NDArray

    # constructor
    def __init__(self, parent, img: npt.NDArray):
        super().__init__(parent)
        self._img = img

        self.figure = plt.figure()

        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        self.accept_button = QPushButton()
        self.accept_button.setText("Accept")
        self.accept_button.clicked.connect(lambda: self.accept())

        self.reject_button = QPushButton()
        self.reject_button.setText("Reject")
        self.reject_button.clicked.connect(lambda: self.reject())

        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        layout.addWidget(self.accept_button)
        layout.addWidget(self.reject_button)
        self.setLayout(layout)

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.imshow(self._img, cmap="bone")
        self.canvas.draw()


# Link button and interface
class PreprocessorPipelineWindow(QDialog):
    _orig_img: npt.NDArray
    _pp: PreprocessorPipeline
    _output_path: Optional[str]

    def __init__(self, ori_img: npt.NDArray):
        self._orig_img = ori_img
        self._pp = PreprocessorPipeline()
        self._output_path = None
        super().__init__()
        self.ui = Ui_MethodSelector()
        self.ui.setupUi(self)
        self.setWindowTitle("PPE <UNSAVED>")

        # Click the button to jump to the corresponding page
        self.ui.pushButton.clicked.connect(lambda: self.ui.stackedWidget.setCurrentIndex(0))
        self.ui.pushButton_2.clicked.connect(lambda: self.ui.stackedWidget.setCurrentIndex(1))
        self.ui.pushButton_3.clicked.connect(lambda: self.ui.stackedWidget.setCurrentIndex(2))
        self.ui.pushButton_4.clicked.connect(lambda: self.ui.stackedWidget.setCurrentIndex(3))
        self.ui.pushButton_17.clicked.connect(lambda: self.ui.stackedWidget.setCurrentIndex(4))
        self.ui.pushButton_18.clicked.connect(lambda: self.ui.stackedWidget.setCurrentIndex(5))
        self.ui.pushButton_19.clicked.connect(lambda: self.ui.stackedWidget.setCurrentIndex(6))
        self.ui.pushButton_20.clicked.connect(lambda: self.ui.stackedWidget.setCurrentIndex(7))
        self.ui.pushButton_21.clicked.connect(lambda: self.ui.stackedWidget.setCurrentIndex(8))
        # close the window
        self.ui.pushButton_24.clicked.connect(lambda: self.close())
        self.ui.pushButton_22.clicked.connect(lambda: self.close())
        self.ui.pushButton_23.clicked.connect(lambda: self.close())
        self.ui.pushButton_25.clicked.connect(lambda: self.close())
        self.ui.pushButton_26.clicked.connect(lambda: self.close())
        self.ui.pushButton_27.clicked.connect(lambda: self.close())
        self.ui.pushButton_28.clicked.connect(lambda: self.close())
        self.ui.pushButton_29.clicked.connect(lambda: self.close())
        self.ui.pushButton_6.clicked.connect(lambda: self.close())
        # Obtaining Parameters
        self.ui.pushButton_7.clicked.connect(self.setparam0)
        self.ui.pushButton_14.clicked.connect(self.setparam1)
        self.ui.pushButton_13.clicked.connect(self.setparam2)
        self.ui.pushButton_12.clicked.connect(self.setparam3)
        self.ui.pushButton_10.clicked.connect(self.setparam4)
        self.ui.pushButton_9.clicked.connect(self.setparam5)
        self.ui.pushButton_8.clicked.connect(self.setparam6)
        self.ui.pushButton_16.clicked.connect(self.setparam7)
        self.ui.pushButton_5.clicked.connect(self.setparam8)

        self.ui.pushButton_15.clicked.connect(lambda: self.save())
        self.ui.pushButton_15.clicked.connect(lambda: self.close())

    def process_kwargs(self, param_dict):
        preprocessor = get_preprocessor(param_dict.pop('name'))()
        try:
            preprocessor = preprocessor.set_params(**param_dict)
        except Exception as e:
            QMessageBox.critical(self, 'ERROR', f"Exception {e} captured!")
            return
        print(f"Preprocessor {repr(preprocessor)}")
        orig_img_copy = self._orig_img.copy()
        transformed_img = self._pp.execute(orig_img_copy)
        try:
            transformed_img = preprocessor.execute(transformed_img)
        except Exception as e:
            QMessageBox.critical(self, 'ERROR', f"Exception {e} captured!")
            return
        plt.imshow(
            transformed_img,
            cmap="bone"
        )
        plt.colorbar()
        plt.show()
        is_accepted = ImageDisplayDialog(
            self,
            transformed_img
        ).exec() == QDialog.Accepted
        if is_accepted:
            self.save(preprocessor)
        else:
            return

    def save(self, preprocessor: Optional[AbstractPreprocessor] = None):
        if self._output_path is None:
            output_path, _ = QFileDialog.getSaveFileName(
                self,
                caption="Select the output path",
                directory=".",
                filter="TOML (*.toml);;ALL (*.*)"
            )
            if output_path == "":
                QMessageBox.critical(
                    self,
                    'ERROR',
                    "Save failed -- You did not select any path!"
                )
                return
            self._output_path = output_path
        self.setWindowTitle(f"PPE <{self._output_path}>")
        if preprocessor is not None:
            self._pp = self._pp.add_step(preprocessor)
        self._pp.save(self._output_path)

    # function of method_1
    def setparam0(self):
        param_dict = {
            'name': 'dumb',
        }
        self.process_kwargs(param_dict)

    # function of method_2
    def setparam1(self):
        param_dict = {
            'name': 'describe',
        }
        self.process_kwargs(param_dict)

    # function of method_3
    def setparam2(self):
        param_dict = {
            'name': 'normalize',
        }
        self.process_kwargs(param_dict)

    # function of method_4
    def setparam3(self):
        param_dict = {
            'name': 'adjust exposure',
        }
        self.process_kwargs(param_dict)

    # function of method_5
    def setparam4(self):
        param_dict = {
            'name': 'denoise (median)',
            'footprint_length_width': self.ui.lineEdit_24.text()
        }
        self.process_kwargs(param_dict)

    # function of method_6
    def setparam5(self):
        param_dict = {
            'name': 'denoise (mean)',
            'footprint_length_width': self.ui.lineEdit_23.text()
        }
        self.process_kwargs(param_dict)

    # function of method_7
    def setparam6(self):
        param_dict = {
            'name': 'denoise (gaussian)',
            'sigma': self.ui.lineEdit_28.text()
        }
        self.process_kwargs(param_dict)

    # function of method_8
    def setparam7(self):
        param_dict = {
            'name': 'unsharp mask',
            'radius': self.ui.lineEdit_31.text(),
            'amount': self.ui.lineEdit.text()
        }
        self.process_kwargs(param_dict)

    # function of method_9
    def setparam8(self):
        param_dict = {
            'name': 'wiener deblur',
            'kernel_size': self.ui.lineEdit_34.text(),
            'balance': self.ui.lineEdit_2.text()
        }
        self.process_kwargs(param_dict)
