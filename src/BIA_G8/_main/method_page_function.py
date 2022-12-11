import matplotlib.pyplot as plt
import numpy.typing as npt
from PyQt5.QtWidgets import QFileDialog, QMainWindow, QMessageBox

from BIA_G8._ui.method_selector import Ui_MethodSelector
from BIA_G8.model.preprocesor_pipeline import PreprocessorPipeline
from BIA_G8.model.preprocessor import get_preprocessor


# Link button and interface
class link_method(QMainWindow):
    _orig_img: npt.NDArray

    def __init__(self, ori_img:npt.NDArray):
        self._orig_img = ori_img
        super().__init__()
        self.ui = Ui_MethodSelector()
        self.ui.setupUi(self)
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
        # dispaly the prompt box
        self.ui.pushButton_7.clicked.connect(self.openwindow3)
        self.ui.pushButton_14.clicked.connect(self.openwindow3)
        self.ui.pushButton_13.clicked.connect(self.openwindow3)
        self.ui.pushButton_12.clicked.connect(self.openwindow3)
        self.ui.pushButton_10.clicked.connect(self.openwindow3)
        self.ui.pushButton_9.clicked.connect(self.openwindow3)
        self.ui.pushButton_8.clicked.connect(self.openwindow3)
        self.ui.pushButton_16.clicked.connect(self.openwindow3)
        self.ui.pushButton_5.clicked.connect(self.openwindow3)
        #
        self.ui.pushButton_11.clicked.connect(self.output_path)
        #
        self.ui.pushButton_15.clicked.connect(self.out)
        #
        self.ui.pushButton_15.clicked.connect(lambda: self.close())


    def openwindow3(self):
        from end_page_function import accept
        self.m = accept()
        self.m.show()

    def Process(self, param_dict):
        global my_pp
        global pp
        pp = PreprocessorPipeline()
        my_pp = get_preprocessor(param_dict.pop('name'))()
        try:
            my_pp = my_pp.set_params(**param_dict)
        except Exception as e:
            QMessageBox.about(self, 'Warning', f"Exception {e} captured!")
        print(f"Preprocessor {repr(my_pp)}")
        orig_img_copy = self._orig_img.copy()
        transformed_img = pp.execute(orig_img_copy)
        try:
            transformed_img = my_pp.execute(transformed_img)
        except Exception as e:
            QMessageBox.about(self, 'Warning', f"Exception {e} captured!")
        plt.imshow(
            transformed_img,
            cmap="bone"
        )
        plt.colorbar()
        plt.show()

    def output_path(self):
        global my_pp
        global pp
        global pp_output_path
        pp_output_path, _ = QFileDialog.getSaveFileName(self, "Select the output path", "\\pyproject", "toml(*.toml)")
        self.ui.lineEdit_3.setText(pp_output_path)

    def out(self):
        global my_pp
        global pp
        global pp_output_path
        pp = pp.add_step(my_pp)
        pp.save(pp_output_path)

    # function of method_1
    def setparam0(self):
        param_dict = {
            'name': 'dumb',
        }
        self.Process(param_dict)

    # function of method_2
    def setparam1(self):
        param_dict = {
            'name': 'describe',
        }
        self.Process(param_dict)

    # function of method_3
    def setparam2(self):
        param_dict = {
            'name': 'normalize',
        }
        self.Process(param_dict)

    # function of method_4
    def setparam3(self):
        param_dict = {
            'name': 'adjust exposure',
        }
        self.Process(param_dict)

    # function of method_5
    def setparam4(self):
        param_dict = {
            'name': 'denoise (median)',
            'footprint_length_width': self.ui.lineEdit_24.text()
        }
        self.Process(param_dict)

    # function of method_6
    def setparam5(self):
        param_dict = {
            'name': 'denoise (mean)',
            'footprint_length_width': self.ui.lineEdit_23.text()
        }
        self.Process(param_dict)

    # function of method_7
    def setparam6(self):
        param_dict = {
            'name': 'denoise (gaussian)',
            'sigma': self.ui.lineEdit_28.text()
        }
        self.Process(param_dict)

    # function of method_8
    def setparam7(self):
        param_dict = {
            'name': 'unsharp mask',
            'radius': self.ui.lineEdit_31.text(),
            'amount': self.ui.lineEdit.text()
        }
        self.Process(param_dict)

    # function of method_9
    def setparam8(self):
        param_dict = {
            'name': 'wiener deblur',
            'kernel_size': self.ui.lineEdit_34.text(),
            'balance': self.ui.lineEdit_2.text()
        }
        self.Process(param_dict)
