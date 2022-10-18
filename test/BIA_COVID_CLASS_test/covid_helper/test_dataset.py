import operator
from functools import reduce

import numpy as np
import pytest
import skimage

from BIA_COVID_CLASS.covid_helper import covid_dataset

_the_image = skimage.img_as_int(np.array(reduce(operator.add, map(lambda x: [[x] * 100] * 10, range(0, 100, 10)))))

_IMAGES = [
    covid_dataset.CovidImage.from_image(_the_image, 0),
    covid_dataset.CovidImage.from_image(_the_image, 0),
    covid_dataset.CovidImage.from_image(_the_image, 1),
    covid_dataset.CovidImage.from_image(_the_image, 1),
    covid_dataset.CovidImage.from_image(_the_image, 2),
]


def test_in_memory_dataset():
    d1 = covid_dataset.CovidDataSet.from_loaded_image(_IMAGES)
    assert d1.data_path == covid_dataset.IN_MEMORY_INDICATOR
    with pytest.raises(ValueError):
        d1.load()
