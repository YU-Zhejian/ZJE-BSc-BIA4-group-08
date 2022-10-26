# Developers' Guide for `proc_profiler`

## Introduction

## Get Started

To participate in development of this project, you need to clone this project to your local computer using [Git](https://git-scm.com) from <https://gitee.com/yuzjlab/2022-23-group-08.git> or <https://github.com/BIA4-course/2022-23-Group-08>.

The Python Coding Standard enforced in this project would be a modified (unpublished) version of [_Google Python Style_](https://google.github.io/styleguide/pyguide.html) (A Chinese version [here](https://zh-google-styleguide.readthedocs.io/en/latest/google-python-styleguide)).

The Developing environment of this project changes rapidly since new machine-learning algorithms \& Implementations are frequently tested, aded or removed. The current "all-in-one" environment is as follows:

% TODO

## Design and Implementation of Contineous Integration

Since this project is small, the automation processes are mainly orchestrated by [GNU Make](https://www.gnu.org/software/make).

```{warning}
The Makefile of this project may not be BSD-make compatible.
```

The documentation and API documentation of the project was built by [Sphinx](https://www.sphinx-doc.org/) with [`myst_parser`](https://myst-parser.readthedocs.io/), which parses MyST-flavored markdown version of documentations and [`nbsphinx`](https://nbsphinx.readthedocs.io/) [^MystNB], which parses executed Jupyter Notebook to Sphinx documentations. Use `make doc` to generate the docuemntation site. the generated documentation site would be available at `doc/_build` related to project root. Use `make cleandoc` to build documentation without cache. Use `make serve-doc` to open documentations on your local browser.

[^MystNB]: We do not use [MyST-NB](https://myst-nb.readthedocs.io) due to its complexity.

The distributable tarball and binary wheel of the project was built with PYPA [`setuptools`](https://setuptools.pypa.io) and PYPA [`build`](https://pypa-build.readthedocs.io). Use `make dist` to generate a distributable [binary wheel](https://packaging.python.org/en/latest/glossary/#term-Wheel) or [source distribution](https://packaging.python.org/en/latest/glossary/#term-Source-Distribution-or-sdist) under `dist` directory.

The unit test of critical infrastructures are set up using [pytest](https://pytest.org). Use `make test` to perform automatic tests and get test reports. The test report together with a JUnit-compatible report in XML and a coverage report would be available at `pytest` directory.

The code of this project is type-annotated, which is recommended for big projects. The project uses [`pytype`](https://google.github.io/pytype) to check type annotations. Use `make pytype` to perform automatic checks.

## Design and Implementation of COVID X-Ray Data Structure

...

## Contribution

Users are welcome to contribute to this project in following aspects:

- **Documentations**
- **Graphical User Interface**
- **Data Structure**
