# Developers' Guide for `proc_profiler`

## Introduction

## Get Started

To participate in development of this project, you need to clone this project to your local computer using [Git](https://git-scm.com) from <https://gitee.com/yuzjlab/2022-23-group-08.git> or <https://github.com/BIA4-course/2022-23-Group-08>.


The Python Coding Standard enforced in this project would be a modified (unpublished) version of [_Google Python Style_](https://google.github.io/styleguide/pyguide.html) (A Chinese version [here](https://zh-google-styleguide.readthedocs.io/en/latest/google-python-styleguide)).

The Developing environment of this project changes rapidly since new machine-learning algorithms \& Implementations are frequently tested, aded or removed. The current "all-in-one" environment is as follows:

## Building the Project

Ensure that you have set up the developmental environment using Conda or Mamba.

Use `make doc` to generate the docuemntation site. the generated documentation site would be available at `doc/_build` related to project root.

Use `make dist` to generate a distributable [binary wheel](https://packaging.python.org/en/latest/glossary/#term-Wheel) or [source distribution](https://packaging.python.org/en/latest/glossary/#term-Source-Distribution-or-sdist).

Use `make test` to perform automatic tests and get test reports. The test report together with a JUnit-compatible report in XML and a coverage report would be available at `pytest` directory.

## Design and Implementation of Project Skeleton

The project skeleton was mainly migrated from several unpublished projects of YU Zhejian.

The documentation of the project was built by [Sphinx](https://www.sphinx-doc.org/) with following extensions:

- [`myst_parser`](https://myst-parser.readthedocs.io/), which parses markdown version of documentations to reStructured Text as is supported by Sphinx.
- [`nbsphinx`](https://nbsphinx.readthedocs.io/), which parses executed Jupyter Notebook to Sphinx documentations.
- [`sphinx_book_theme`](https://sphinx-book-theme.readthedocs.io/), the Sphinx HTML theme.

## Design and Implementation of COVID X-Ray Data Structure
