import numpy as np
import pytest
import torch

from BIA_KiTS19.helper import ndarray_helper

_test_cases = {
    "1d_1": [
        {"array": np.array([1, 2, 3, 4]), "axis": 0, "start": 1, "end": 3},
        np.array([2, 3])
    ],
    "1d_2": [
        {"array": np.array([1, 2, 3, 4]), "start": 1, "end": 3},
        np.array([2, 3])
    ],
    "1d_3": [
        {"array": np.array([1, 2, 3, 4]), "start": 1},
        np.array([2, 3, 4])
    ],
    "2d_1": [
        {"array": np.array([
            [1, 2, 3, 4],
            [5, 6, 7, 8]
        ]), "start": 1},
        np.array([[5, 6, 7, 8]])
    ],
    "2d_2": [
        {"array": np.array([
            [1, 2, 3, 4],
            [5, 6, 7, 8]
        ]), "axis": 1, "start": 1},
        np.array([[2, 6], [3, 7], [4, 8]])
    ]
}


@pytest.mark.parametrize(
    argnames="kwargs",
    argvalues=_test_cases.values(),
    ids=_test_cases.keys()
)
def test_sample_along_np(kwargs):
    assert np.array_equiv(ndarray_helper.sample_along_np(**kwargs[0]), kwargs[1])


@pytest.mark.parametrize(
    argnames="kwargs",
    argvalues=_test_cases.values(),
    ids=_test_cases.keys()
)
def test_sample_along_tensor(kwargs):
    kwargs[0]["array"] = torch.tensor(kwargs[0]["array"])
    assert torch.equal(ndarray_helper.sample_along_tensor(**kwargs[0]), torch.tensor(kwargs[1]))
