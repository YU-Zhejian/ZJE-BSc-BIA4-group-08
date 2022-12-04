import os
import platform
import shutil
import tempfile

import numpy as np
import pytest
import skimage
import skimage.transform as skitrans

from BIA_G8_DATA_ANALYSIS import covid_dataset
from BIA_G8_DATA_ANALYSIS_test.covid_helper import stride

label = "LBL"


def test_apply():
    c1 = covid_dataset.CovidImage.from_np_array(stride, 0, label)
    c2 = c1.apply(lambda img: skitrans.rotate(img, 90))
    assert c2.label == 0
    assert c2.label_str == label
    c3 = c2.apply(lambda img: skimage.img_as_int(skitrans.rotate(img, 270)))
    assert np.array_equiv(c1.np_array, c3.np_array)


def test_save_load():
    c1 = covid_dataset.CovidImage.from_np_array(stride, 100, label)
    assert c1.label == 100
    assert c1.label_str == label
    assert c1.image_path == covid_dataset.IN_MEMORY_INDICATOR
    tmp_dir = tempfile.mkdtemp()
    os.makedirs(os.path.join(tmp_dir, label))

    tmp_save_path_npy = os.path.abspath(os.path.join(tmp_dir, label, "1.npy.xz"))
    c1.save(tmp_save_path_npy)
    assert c1.image_path == tmp_save_path_npy
    c2 = covid_dataset.CovidImage.from_file(tmp_save_path_npy, 100, label)
    assert c2.image_path == tmp_save_path_npy
    assert c2.label == 100
    assert c2.label_str == label
    assert np.array_equiv(c1.np_array, c2.np_array)
    c2.save()

    tmp_save_path_tiff = os.path.abspath(os.path.join(tmp_dir, label, "1.tiff"))
    c1.save(tmp_save_path_tiff)
    assert c1.image_path == tmp_save_path_tiff
    c3 = covid_dataset.CovidImage.from_file(tmp_save_path_tiff, 100, label)
    assert c3.image_path == tmp_save_path_tiff
    assert c3.label == 100
    assert c3.label_str == label
    assert np.array_equiv(c1.np_array, c3.np_array)

    shutil.rmtree(tmp_dir)
    with pytest.raises(ValueError):
        c2.save(covid_dataset.IN_MEMORY_INDICATOR)


def test_resolve_label_from_path():
    if platform.system() != "Windows":
        assert covid_dataset.resolve_label_str_from_path("/100/200/300.png") == "200"
    else:
        assert covid_dataset.resolve_label_str_from_path("D:\\100\\200\\300.png") == "200"
