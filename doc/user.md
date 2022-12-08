# Users' Guide for `BIA_G8`

## Concepts and Procedures


## Recommended Runtime Environment

- You are recommended to use this software on a Workstation with recent x86\_64 (Intel/AMD) CPU and 16 GiB of memory. Using this software on domestic laptops is possible but slow.
- You are recommended to use POSIX-compiliant operating system.  Microsoft Windows is supported but with impaired performance.
- If you wish to use pre-processors or classifiers with neuron networks (e.g., SCGAN, ToyCNN, ResNet50), you are recommended to NVidia General-Purposed Graph Processing Unit (GPGPU) with 6 GiB GDDR6 display memory (e.g., NVidia GeForce RTX 2060). Using CPU is possible but much slower.
- Your operating system should have a Python intepreter (version at least 3.8; CPython implementation). You are recommended to manage the dependencies using [Conda](https://docs.conda.io/), which is a general-purposed user-level package management system.

## Usage with Installation

This software can be installed in following ways:

1. Use pre-built binary wheels. Install this software using `pip install /path/to/BIA-G8-XXX.whl`.
2. Clone this repository, build the wheels using `setuptools` and install it.
    1. Clone this repository using [Git](https://git-scm.com). We suppose you cloned it into somewhere called `{PROJECT_ROOT}`. Command: `git clone https://gitee.com/yuzjlab/2022-23-group-08`.
    2. Install `setuptool`, `build` and `pip`.
    3. Build the wheel using `python -m build`.
    4. Install the wheel built at `{PROJECT_ROOT}/dist`.

### Using the Commandline Interface

Use command-line version of Preprocessor Explorer using:

```shell
python -m BIA_G8._main.preprocessor_explorer --help
```

% TODO

### Using the Graphical User Interface

% TODO

````{hint}
**Always check you're using the correct Python Intepreter.**

Windows CMD:

```bat
where python
```

and look at the first value.

Windows Powershell:

```powershell
cmd /C where python
```

or:

```powershell
Get-Command python
```

POSIX shell with GNU CoreUtils (or similiar alternative):

```shell
which python
```

If the Python intepreter displayed is not what you want, you may retry by replacing `python` with `python3` or `py`.

Conda may not work well on Windows Powershell, so using CMD is recommended.

Conda may forgot to update the `PATH` environment variable after activating corresponding environment. If so, you may ivoke Python using `${CONDA_PREFIX}/python`.
````

## Tutorials on Data Structures

Following pages contains tutorials on data structures used in this project. You may use them on your own datasets.


```{toctree}
:glob:
:maxdepth: 2

ipynb/data_structure
ipynb/ml_pp
```
