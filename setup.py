"""``setup.py`` that is replaced by ``pyproject.toml``"""
from setuptools import setup

from compile_ui import compile_all_ui, compile_all_rc

if __name__ == '__main__':
    compile_all_ui()
    compile_all_rc()
    setup()
