# Developers' Guide for `BIA_G8`

## Get Started

To participate in the development of this project, you need to clone this project to your local computer using [Git](https://git-scm.com) from <https://gitee.com/yuzjlab/2022-23-group-08.git> or <https://github.com/BIA4-course/2022-23-Group-08>.

The Python Coding Standard enforced in this project would be [_Google Python Style_](https://google.github.io/styleguide/pyguide.html) (A Chinese version [here](https://zh-google-styleguide.readthedocs.io/en/latest/google-python-styleguide)).

The Developing environment of this project changes rapidly since new machine-learning algorithms \& Implementations are frequently tested, added, or removed. Set up the development environment using [Conda](https://docs.conda.io/en/latest/) with configuration files at `{PROJECT_ROOT}/envs`. You may also use PIP-based virtual environment providers like [venv](https://docs.python.org/3/library/venv.html), [pipenv](https://pipenv.pypa.io/en/latest/index.html) or [virtualenv](https://virtualenv.pypa.io) with both `{PROJECT_ROOT}/requirements.txt` and `{PROJECT_ROOT}/requirements-dev.txt`.

```{hint}
Since the environment is large, it may take tens of minutes for Conda to resolve the dependencies. You are recommended to use [mamba](https://mamba.readthedocs.io/) as a drop-in replacement for Conda, which is faster.
```

Before running the applications in development mode, you may activate the environment and add the `src` directory to the `PYTHONPATH` environment variables.

Using CMD:

```bat
rem Go to the partition if your working directory is not on the same partition with {PROJECT_ROOT}
{X}:
cd {PROJECT_ROOT}
conda activate BIA4Env
set PYTHONPATH=src
```

Using Bash:

```bash
cd {PROJECT_ROOT}
conda activate BIA4Env
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"
```

## Design and Implementation of Contineous Integration

Since this project is small, the automation processes are mainly orchestrated by [GNU Make](https://www.gnu.org/software/make).

```{warning}
The Makefile of this project may not be BSD-make compatible.
```

The documentation and API documentation of the project was built by [Sphinx](https://www.sphinx-doc.org/) with [`myst_parser`](https://myst-parser.readthedocs.io/), which parses MyST-flavored markdown version of documentations and [MyST-NB](https://myst-nb.readthedocs.io), which parses executed Jupyter Notebook to Sphinx documentations. Use `make doc` to generate the documentation site. the generated documentation site would be available at `doc/_build` related to the project root. Use `make cleandoc` to build documentations without a cache. Use `make serve-doc` to open the documentations on your local browser.

The distributable tarball and binary wheel of the project were built with PYPA [`setuptools`](https://setuptools.pypa.io) and PYPA [`build`](https://pypa-build.readthedocs.io). Use `make dist` to generate a distributable [binary wheel](https://packaging.python.org/en/latest/glossary/#term-Wheel) or [source distribution](https://packaging.python.org/en/latest/glossary/#term-Source-Distribution-or-sdist) under `dist` directory.

The unit test of critical infrastructures was set up using [pytest](https://pytest.org). Use `make test` to perform automatic tests and get test reports. The test report together with a JUnit-compatible report in XML and a coverage report would be available in `pytest` directory.

The code of this project is type-annotated, which is recommended for big projects. The project uses [`pytype`](https://google.github.io/pytype) to check type annotations. Use `make pytype` to perform automatic checks.

## Design and Implementation of Abstract Interface

### Abstract Dataset Interface

The abstract dataset interface, {py:class}`BIA_G8.helper.ml_helper.MachinelearningDatasetInterface`, is a simple interface that supports being trained and evaluated by {py:mod}`sklearn` and {py:mod}`torch`. with a {py:func}`sklearn.model_selection.train_test_split()`-like method to split itself into a training set and a testing set.

It have one child class {py:class}`BIA_G8.data_analysis.covid_dataset.CovidDataSet`. See below for more details.

### Pre-Processors and Pre-Processing Pipelines

Pre-processors are functions that take an image as input and return a processed image to assist the classifier in making better classifications, and the pre-processing pipeline is an ordered list of pre-processors that sequentially execute each processor on an input image.

The pre-processor, whose abstract type is defined in {py:class}`BIA_G8.model.preprocessor.AbstractPreprocessor`, is a general-purpose pre-processor interface that supports arguments of different types.

The pre-processing pipelines can be serialized to and deserialized from TOML files. User can create their own pre-processing pipeline using `preprocessor_explorer`, whose workload is as follows:

1. The frontend (`preprocessor_explorer`) query available pre-processor name and description from {py:func}`BIA_G8.model.preprocessor.get_preprocessor_name_descriptions()`, and user select wanted pre-processor class name.

    ```{note}
    In a reflex-like pathway, the name and description are class attributes instead of instance attributes.
    ```

2. The pre-processor class is instantiated using the reflex-like pathway ({py:func}`BIA_G8.model.preprocessor.get_preprocessor`), and the argument names and descriptions are queried by the frontend using read-only attribute {py:data}`BIA_G8.model.preprocessor.AbstractPreprocessor.arguments`.

    ```{note}
    **Abstraction of arguments.**

    Argument parsers are represented as {py:class}`BIA_G8.model.preprocessor.AbstractPreprocessor`. They are nullable argument processors and those who use this argument parser should check whether the output is {py:data}`BIA_G8.model.unset`.
    ```

3. After the user input their arguments, they are validated by {py:func}`BIA_G8.model.preprocessor.AbstractPreprocessor.set_params` and stored.
4. The final execution would be done by `BIA_G8.model.preprocessor.AbstractPreprocessor.execute`, which uses stored arguments.

### Classifiers

Since this project employ traditional machine- and deep-learning based methods using different API [^sklearn], there is a need for an abstraction layer that unifies them.

[^sklearn]: There is also a problem with typing systems in {py:mod}`sklearn`. It uses Maxin instead of the traditional inheritance-based OOP programming mode, making it hard to extend.

The classifier abstract class, {py:class}`BIA_G8.model.classifier.ClassifierInterface`, is a general-purpose classifier that supports training, evaluation, and predicting. The interface also supports loading and saving pre-trained models and parameters.

Referenced: [skorch](https://skorch.readthedocs.io/en/stable), which is a wrapper that wraps {py:mod}`torch` API to {py:mod}`sklearn`-compatible API.

## Design and Implementation of COVID X-Ray Data Structure

The COVID Data structure ({py:class}`BIA_G8.data_analysis.covid_dataset.CovidDataSet`) is an ordered list of X-Ray Images ({py:class}`BIA_G8.data_analysis.covid_dataset.CovidImage`) that can be represented as Numpy NDArray or {py:class}`torch.Tensor`.

The data structure is optimizied for fast and secure in-memory analytics, which requires the following aspects:

- All objects are immutable. This is implemented by setting all attributes to read-only, turning lists into iterables, and setting writable flags to numpy arrays.
- Supports parallel operation. This is implemented by the embarrassinglly parallel implementation provided by {py:mod}`joblib`.
- Need no on-disk caching mechanism.

The dataset supports parallel applications of operations on each image, making it suitable for explorational tasks. The dataset also supports sampling, loading images from disk, and saving them to disk.

The machine-learning mechanism also requires an encoder and a decoder, which are passed into the class as functions. The data structure can also infer the encoder-decoder pair from  the file system.

## Design and Implementation of Grid Search Engine

From the unified APIs, we can design a very simple grid search engine. The grid search engine, as is implemented in {}, performs grid search over a series of pre-processor pipeline configurations, classifier configurations, and dataset configurations.
