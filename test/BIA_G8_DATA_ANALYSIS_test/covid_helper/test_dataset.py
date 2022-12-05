import shutil
import tempfile

import numpy as np
import pytest
import skimage
import skimage.transform as skitrans

from BIA_G8.helper import ml_helper
from BIA_G8_DATA_ANALYSIS import covid_dataset
from BIA_G8_DATA_ANALYSIS_test.covid_helper import stride

_DEFAULT_ENCODER_DICT = {
    "NA": 100,
    "COVID-19": 0,
    "NORMAL": 1,
    "Viral_Pneumonia": 2
}
encode, decode = ml_helper.generate_encoder_decoder(_DEFAULT_ENCODER_DICT)


def create_in_memory_images():
    imgs = []
    for label, img in enumerate([stride] * 3):
        imgs.append(covid_dataset.CovidImage.from_np_array(img, label, decode(label)))
        imgs.append(covid_dataset.CovidImage.from_np_array(img, label, decode(label)))

    _ = imgs.pop()
    return imgs


def test_in_memory_dataset():
    d1 = covid_dataset.CovidDataSet.from_loaded_images(
        create_in_memory_images(),
        encode=encode,
        decode=decode,
        n_classes=3
    )
    assert d1.dataset_path == covid_dataset.IN_MEMORY_INDICATOR
    d2 = d1.sample(balanced=True)
    assert len(d2) == 3
    with pytest.raises(ValueError):
        _ = d1.sample(size=4, balanced=True)
    with pytest.raises(ValueError):
        _ = d1.sample(size=9, balanced=False)
    d3 = d1.sample(size=2, balanced=False)
    assert len(d3) == 2
    d4 = d1.sample(size=2, balanced=True)
    assert len(d4) == 0
    d5 = d1.sample(balanced=False)
    assert len(d5) == 5
    d6 = d1.sample()
    assert len(d6) == 3


def test_apply():
    d1 = covid_dataset.CovidDataSet.from_loaded_images(
        create_in_memory_images(),
        encode=encode,
        decode=decode,
        n_classes=3
    )
    d2 = d1.apply(lambda img: skimage.img_as_int(skitrans.rotate(img, 270))).apply(
        lambda img: skimage.img_as_int(skitrans.rotate(img, 90)))
    for i1, i2 in zip(d1, d2):
        assert np.array_equiv(i1.np_array, i2.np_array)


def test_parallel_apply():
    d1 = covid_dataset.CovidDataSet.from_loaded_images(
        create_in_memory_images(),
        encode=encode,
        decode=decode,
        n_classes=3
    )
    d2 = d1.apply(lambda img: skimage.img_as_int(skitrans.rotate(img, 270))).parallel_apply(
        lambda img: skimage.img_as_int(skitrans.rotate(img, 90)))
    for i1, i2 in zip(d1, d2):
        assert np.array_equiv(i1.np_array, i2.np_array)


def test_load_save_dataset():
    tmp_dir = tempfile.mkdtemp()

    d1 = covid_dataset.CovidDataSet.from_loaded_images(
        create_in_memory_images(),
        encode=encode,
        decode=decode,
        n_classes=3
    )
    d1.save(tmp_dir)
    assert d1.dataset_path == tmp_dir
    d2 = covid_dataset.CovidDataSet.from_directory(
        tmp_dir,
        balanced=False,
        encode=encode,
        decode=decode
    )
    assert d2.dataset_path == tmp_dir
    assert len(d1) == len(d2)
    for i1, i2 in zip(d1, d2):
        assert np.array_equiv(i1.np_array, i2.np_array)

    d1.save(tmp_dir, extension="png")
    assert d1.dataset_path == tmp_dir
    d2 = covid_dataset.CovidDataSet.from_directory(
        tmp_dir,
        balanced=False,
        encode=encode,
        decode=decode
    )
    assert d2.dataset_path == tmp_dir
    assert len(d1) == len(d2)
    for i1, i2 in zip(d1, d2):
        assert np.array_equiv(i1.np_array, i2.np_array)
    shutil.rmtree(tmp_dir)


def test_parallel_load_save_dataset():
    tmp_dir = tempfile.mkdtemp()
    d1 = covid_dataset.CovidDataSet.from_loaded_images(
        create_in_memory_images(),
        encode=encode,
        decode=decode,
        n_classes=3
    )
    d1.parallel_save(tmp_dir)
    assert d1.dataset_path == tmp_dir
    d2 = covid_dataset.CovidDataSet.parallel_from_directory(
        tmp_dir,
        encode=encode,
        decode=decode,
        n_classes=3
    )
    assert d2.dataset_path == tmp_dir
    for i1, i2 in zip(d1, d2):
        assert np.array_equiv(i1.np_array, i2.np_array)
    shutil.rmtree(tmp_dir)
