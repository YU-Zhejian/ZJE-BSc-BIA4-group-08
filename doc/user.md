# Users' Guide for `BIA_G8`

## Concepts and Procedures

- An **image** is a two-dimensional, grey-scale, `skimage`-parsable (i.e., JPG, PNG, TIFF, etc.. DICOM/NIFFT is not parsale in current version) chest X-ray image.
- A **classifier** classifies image to COVID, Normal and Viral Pneumonia.
- A **preprocessor** transforms the image in a certain way that assists the classifier to classify the image.
- A **preprocessor pipeline** consists of some preprocessors that are connected together. They transform the image in a sequential way.
- **TOML** is the format in which configurations and are saved.

## Recommended Runtime Environment

- You are recommended to use this software on a Workstation with a recent x86\_64 (Intel/AMD) CPU and 16 GiB of memory. Using this software on domestic laptops is possible but slow.
- You are recommended to use a POSIX-compiliant operating system.  Microsoft Windows is supported but with impaired performance.
- If you wish to use pre-processors or classifiers with neuron networks (e.g., SCGAN, ToyCNN, ResNet50), you are recommended to NVidia General-Purposed Graph Processing Unit (GPGPU) with 6 GiB GDDR6 display memory (e.g., NVidia GeForce RTX 2060). Pre-trained models in CPU is not provided since training on CPU is too slow.
- Your operating system should have a Python interpreter (version at least 3.8; CPython implementation). You are recommended to manage the dependencies using [Conda](https://docs.conda.io/), which is a general-purpose user-level package management system.

````{hint}
**Always check whether you are using the correct Python Interpreter.**

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

Conda may forget to update the `PATH` environment variable after activating the corresponding environment. If so, you may invoke Python using `${CONDA_PREFIX}/python`.
````

## Installation

This software can be installed in the following ways:

1. Use pre-built binary wheels. Install this software using `pip install /path/to/BIA-G8-XXX.whl`.

    ```{note}
    % TODO
    For BIA4 marking: The binary wheel is available at <here>.
    ```

2. Clone this repository, build the wheels using `setuptools`, and install it.
    1. Clone this repository using [Git](https://git-scm.com). We suppose you cloned it into somewhere called `{PROJECT_ROOT}`. Command: `git clone https://gitee.com/yuzjlab/2022-23-group-08`.
    2. Install `setuptool`, `build`, and `pip`.
    3. Build the wheel using `python -m build`.
    4. Install the wheel built at `{PROJECT_ROOT}/dist`.

```{note}
% TODO
For BIA4 marking: You may use pretrained data available at <https://zjuintl-my.sharepoint.com/:f:/g/personal/zhejian_19_intl_zju_edu_cn/Ej931YIXtr1EmYsDXLs7QW8B0pby-0XIzyv3kE1nP8PRUg?e=TUIxOg>.
```

## Preprocesor Explorer (PPE): CLI

Before, we need to create the preprocessor configuration file: in TOML format. This step can be done by Preprocessor Explorer, or use pre-trained preprocessing configurations available at `{PROJECT_ROOT}/pretrained/pp_*.toml`.

An example of preprocessor pipeline config:

```toml
[0]
name = "dumb"

[0.args]

[1]
name = "normalize"

[1.args]

[version_info]
BIA_G8 = "0.0.1"
numpy = "1.23.4"
sklearn = "1.1.3"
joblib = "1.1.1"
xgboost = "1.7.0"
matplotlib = "3.6.2"
skimage = "0.19.3"
tomli = "2.0.1"
tomli-w = "1.0.0"
torch = "1.13.0"
python = "3.8.13"

[metadata]
time_gmt = "Sun Dec 11 07:29:12 2022"
time_local = "Sun Dec 11 15:29:12 2022"

[metadata.platform_uname]
system = "Windows"
node = "DESKTOP-CGV4O9H"
release = "10"
version = "10.0.25236"
machine = "AMD64"
processor = "Intel64 Family 6 Model 142 Stepping 12, GenuineIntel"
```

This preprocessor pipeline contains 2 steps: The first step does nothing and the second step normalizes the image to required format.

You may also use interactive Preprocessor Explorer (PPE) where you can add step, observe its effect on a sample image.

Use command-line version of PPE using:

```shell
python -m BIA_G8._main.preprocessor_explorer --input_path {SAMPLE_IMG} --pp_output_path {OUTPUT_TOML}
```

where `{SAMPLE_IMG}` is the sample chest X-ray image you wish to use and `{OUTPUT_TOML}` is the destination TOML file. After starting the PPE, you would see an interface like:

```text
Preprocessor Ready. Available Preprocessors:
        0: dumb -- This preprocessor does nothing!
        1: describe -- This preprocessor describes status of current image
        2: normalize -- This preprocessor normalize the image for analysis
        3: adjust exposure -- This preprocessor can correct the image if it is underexposed or overexposed
        4: denoise (median) -- This preprocessor can remove noise, especially salt & pepper noise
        5: denoise (mean) -- This preprocessor can smooth the image and remove unnecessary details
        6: denoise (gaussian) -- This preprocessor can remove noise, especially gaussian noise
        7: unsharp mask -- This preprocessor can sharp the images, not recommended unless the image is blur
        8: wiener deblur -- This preprocessor can deblur the images which is caused by the mild movement of patient
        9: SCGAN deblur -- This preprocessor can deblur the images using SCGAN
Enter Preprocessor ID (-1 to exit) >
```

The PPE would print all available pre-processors and its description, and the user should select one using the index displayed before the name. Use `-1` to save and exit PPE.

For example, we select `2` and press enter. PPE would print:

```text
Selected preprocessor normalize.
Preprocessor Initialized filter 'normalize' with arguments {}
```

The argument `{}` indicates that this pre-processor does not need any argument. After displaying above image, PPE would show image with this pre-processor applied. You can then choose whether to add this pre-processor to the pipeline by entering `Y` to the commandline interface or `N` to discard this pre-processor. PPE would save the pipeline once a `Y` is entered.

There are also pre-processors that accepts arguments. For example, `8` -- `wiener deblur` requires 2 arguments looked like this:

```text
Selected preprocessor wiener deblur.
Argument ``kernel_size`` (required: True) -- the point distribution pattern, recommended range: integer from 2 to 5
Argument Value for kernel_size (blank for default) >2
Argument ``balance`` (required: True) -- a parameter that tune the data restoration, recommended range: float from 0.005 to 1
Argument Value for balance (blank for default) >0.1
Preprocessor Initialized filter 'wiener deblur' with arguments {"kernel_size": 2, "balance": 0.1}
Accept? Y/N >Y
```

If an exception is caught, PPE would discard the pre-processor and return to the initial state. For example:

```text
Enter Preprocessor ID (-1 to exit) >7
Selected preprocessor unsharp mask.
Argument ``radius`` (required: False) -- the larger it is, the clearer the edges, recommended range: float from 1 to 5
Argument Value for radius (blank for default) >-1
Argument ``amount`` (required: False) -- the extent that the detail is amplified, recommended range: float from 1 to 15
Argument Value for amount (blank for default) >-1
Preprocessor Initialized filter 'unsharp mask' with arguments {"radius": -1.0, "amount": -1.0}
Exception Sigma values less than zero are not valid captured!
```

## Preprocesor Explorer (PPE): GUI

Please execute `python -m BIA_G8._main.preprocessor_explorer_gui`, and a GUI should be displayed.

1. Click "Upload" to input the picture from yur local PC.

    ![LEGEND](figs/upload.png)

2. Click "Start" to select the preprocessing method.

    ![LEGEND](figs/start.png)

3. Apply your preprocessing method to your picture, and the procedures are shown as below: 
    1. Click "method_3" and click "OK", then the normalized picture will be showed up like this:

        ![LEGEND](figs/method-3.png)
        ![LEGEND](figs/accept1.png)

        Click "Accept" to receive the modification and you can save the new picture to your local PC. 

        ![LEGEND](figs/save.png)

    2. Apply suitable preprocessing methods and parameters to your normalized picture. Here, we take the example of clicking "method_8" ('Unsharp mask'): Click "method_8" and set the radius at 5, amount at 5, then click "OK" to apply these modifications.

        ![LEGEND](figs/method-8.png)

        After that, the new preprocessed picture will be showed up. Click "Accept" and the new modified picture will cover the old version. 

        ![LEGEND](figs/accept2.png)

    3. Click "method_3" and "OK", then click "Accept" to to normalize the modified picture again. This modification will cover the old version as well. 

        ![LEGEND](figs/accept3.png)

4. Click " Save Exit" to exit the software and complete the preprocessing steps.

    ![LEGEND](figs/save_exit.png)

## Predictor: CLI

Please execute `python -m BIA_G8._main.perform_ml  --preprocessor_pipeline_config_path {PP.TOML} --classifier_configuration_path {CLASS.TOML} --input_image_path {IMAGES} --out_csv {OUT.CSV}`, where `{PP.TOML}` is the preprocessor pipeline configuration generated by PPE, `{CLASS.TOML}` is a file nearby pre-trained model (should be named like `{CLASS.TOML}.model.pth.xz`), `{IMAGES}` is the path to a directory containing images and `{OUT.CSV}` is where the output would be saved.

## Predictor: GUI

Please execute `python -m BIA_G8._main.perform_ml_gui`, and a GUI should be displayed.

![LEGEND](figs/predictor_gui_menu.png)

In the predictor, you need to open the folder containing images (which would be recursively scanned), the preprocessor pipeline configuration generated by PPE and machine learning model in TOML format. They you can click `Execute` in `Run` menu and the classification would be done. You can save the result onto disk using CSV format.

## Dataset Explorer (Experimental, Not Finished)

We also have a dataset explorer that may assist data scientists to perform quality conyrol on their training set before feeding them to machine-learning pipelines. Currently it can only produce a TSNE plot without any aditional function. You may use `python -m BIA_G8._main.data_explorer` and/or `python -m BIA_G8._main.data_explorer_gui` to have it tested.

Functions on dimension, format and range detection are purposed.

## Tutorials on Data Structures

The following pages contain tutorials on data structures used in this project. You may use them on your own datasets.

```{toctree}
:glob:
:maxdepth: 2

ipynb/data_structure
ipynb/ml_pp
```
