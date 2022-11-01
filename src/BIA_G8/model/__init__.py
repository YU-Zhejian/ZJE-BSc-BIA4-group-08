from torch import nn

from BIA_G8 import get_lh
from BIA_G8.helper import ndarray_helper

_lh = get_lh(__name__)


class Describe(nn.Module):
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
        self.describe = lambda x: prefix + ndarray_helper.describe(x)

    def forward(self, x):
        """"""
        _lh.debug(self.describe(x))
        return x
