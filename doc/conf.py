# ==============================================================================
#  Copyright (C) 2021-2022. tetgs authors
#
#  This file is a part of tetgs, which is licensed under MIT,
#  a copy of which can be obtained at <https://opensource.org/licenses/MIT>.
#
#  NAME: conf.py -- Configure file of Sphinx.
#
#  VERSION HISTORY:
# 2021-08-20 0.1  : Purposed and added by YU Zhejian.
#
# ==============================================================================
"""
Configuration file for the Sphinx documentation builder.
"""

import glob
import os
import pkgutil
import shutil
import sys

import tomli

os.environ['SPHINX_BUILD'] = '1'  # Disable chronolog.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(THIS_DIR)

# Enable scan of packages in src which is not installed.
sys.path.insert(0, os.path.join(ROOT_DIR, 'src'))

import BIA_G8


def scan_dir(path_to_scan: str):
    """
    Recursively list files in directories.

    <https://www.sethserver.com/python/recursively-list-files.html>
    """

    files = []
    dirlist = [path_to_scan]
    while len(dirlist) > 0:
        for (dirpath, dirnames, filenames) in os.walk(dirlist.pop()):
            dirlist.extend(dirnames)
            files.extend(map(lambda n: os.path.join(*n), zip([dirpath] * len(filenames), filenames)))
    return files


def copy_doc_files(from_path: str, to_path: str):
    """
    Copy items to project root
    """
    os.makedirs(to_path, exist_ok=True)
    for name in glob.glob(from_path):
        shutil.copy(name, to_path + os.sep)


copy_doc_files(os.path.join(ROOT_DIR, '*.md'), os.path.join(THIS_DIR, "_root"))

# -- Project information -----------------------------------------------------

with open(os.path.join(ROOT_DIR, "pyproject.toml"), "rb") as reader:
    parsed_pyproject = tomli.load(reader)

project = parsed_pyproject["project"]["name"]
author = "&".join([author["name"] for author in parsed_pyproject["project"]["authors"]])
copyright_string = f'2022, {author}'
release = BIA_G8.__version__

# -- General configuration ---------------------------------------------------

# The theme to use for HTML and HTML Help pages.
html_theme = 'sphinx_book_theme'

# Sphinx extensions
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.todo',
    'sphinx.ext.mathjax',
    'rst2pdf.pdfbuilder',
    'myst_parser',
    html_theme
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The sphinxignore.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    '_build',
    'Thumbs.db',
    '.DS_Store',
    '.virtualenv/**'
]

# -- Options for HTML output -------------------------------------------------
html_theme_options = {
    'show_navbar_depth': 2,
    'repository_branch': "master",
    "home_page_in_toc": True,
    "toc_title": "Page Table of Contents",
    "use_download_button": True,
    "use_repository_button": True,
    "use_issues_button": True,
}

# Copy theme HTMLs. A bug in sphinx_book_theme
theme_loader = pkgutil.get_loader(html_theme)
if theme_loader is None:
    raise FileNotFoundError(f"Cannot load theme {html_theme}!")
theme_path = os.path.join(os.path.dirname(theme_loader.path), "_templates")
THIS_TEMPLATE = os.path.join(THIS_DIR, "_templates")
if not os.path.exists(THIS_TEMPLATE):
    shutil.copytree(theme_path, THIS_TEMPLATE)

# html_logo = "_static/logo.svg"  # Logo at top left of the page.
# html_favicon = "_static/logo.svg"  # Logo at opened tabs.

# Override built-in static files.
# html_static_path = ['_static']

# Insert both docstring of the class and constructor.
# autodoc_default_options = {
#     'special-members': '__init__',
# }

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

latex_elements = {
    'maxlistdepth': '20',
}

# Insert both docstring of the class and constructor.
autodoc_default_options = {
    'special-members': '__init__',
}
