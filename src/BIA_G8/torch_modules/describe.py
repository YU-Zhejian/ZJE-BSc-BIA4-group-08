import torch

from BIA_G8.helper import ndarray_helper
from BIA_G8.torch_modules import AbstractTorchModule

from BIA_G8 import get_lh

_lh = get_lh(__name__)


class Describe(AbstractTorchModule):
    """
    The Describe Layer of PyTorch Module.

    Prints the description of matrix generated from last layer and pass the matrix without modification.
    """

    def __init__(self, prefix: str = ""):
        """
        The initializer

        :param prefix: Prefix of the printed message. Recommended to be the name of previous layer.

        See also: py:func:`BIA_G8.helper.ndarray_helper.describe`.
        """
        super().__init__()
        self._prefix = prefix

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """"""
        _lh.debug(self._prefix + ndarray_helper.describe(x))
        return x
