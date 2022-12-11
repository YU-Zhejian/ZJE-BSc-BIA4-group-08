import glob
import os

from PyQt5.uic import compileUi


def compile_all_ui():
    for ui_fn in glob.glob(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "src",
                "BIA_G8",
                "_ui",
                "*.ui"
            )
    ):
        py_fn = ui_fn.replace(".ui", ".py")
        with open(py_fn, "w") as pyfile:
            compileUi(
                uifile=ui_fn,
                pyfile=pyfile
            )


if __name__ == '__main__':
    compile_all_ui()
