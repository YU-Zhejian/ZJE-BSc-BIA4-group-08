import os

import numpy as np
import pytest
import torch

from BIA_KiTS19.helper import io_helper

_test_cases = {
    "0d": 1.1,
    "1d": np.array([0.1, 1.1]),
    "2d": np.array([[0.1, 1.1], [1.2, 5.0]]),
    "inf": np.array(np.inf),
    # "nan": np.array(np.nan), # Which would fail
    "pi": np.array(np.pi)
}


@pytest.mark.parametrize(
    argnames="kwargs",
    argvalues=_test_cases.values(),
    ids=_test_cases.keys()
)
def test_np_xz(kwargs):
    io_helper.save_np_xz(kwargs, "tmp.xz")
    assert np.array_equiv(io_helper.read_np_xz("tmp.xz"), kwargs)
    os.remove("tmp.xz")


@pytest.mark.parametrize(
    argnames="kwargs",
    argvalues=_test_cases.values(),
    ids=_test_cases.keys()
)
def test_tensor_xz(kwargs):
    kwargs = torch.tensor(kwargs)
    io_helper.save_tensor_xz(kwargs, "tmp.xz")
    assert torch.equal(io_helper.read_tensor_xz("tmp.xz"), kwargs)
    os.remove("tmp.xz")
