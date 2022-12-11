#!/usr/bin/env python
import glob
import os

from PyQt5.uic import compileUi
from PyQt5.pyrcc_main import processResourceFile


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
        print(f"{ui_fn} -> {py_fn}")
        with open(py_fn, "w") as pyfile:
            compileUi(
                uifile=ui_fn,
                pyfile=pyfile
            )


def compile_all_rc():
    for rcc_fn in glob.glob(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "src",
                "BIA_G8",
                "_ui",
                "*.qrc"
            )
    ):
        py_fn = rcc_fn.replace(".qrc", "_rc.py")
        print(f"{rcc_fn} -> {py_fn}")
        processResourceFile(
            filenamesIn=[rcc_fn],
            filenameOut=py_fn,
            listFiles=False
        )


if __name__ == '__main__':
    compile_all_ui()
    compile_all_rc()
