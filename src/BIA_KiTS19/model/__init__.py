from torch import nn

from BIA_KiTS19 import get_lh
from BIA_KiTS19.helper import ndarray_helper

_lh = get_lh(__name__)


class Describe(nn.Module):
    def __init__(self, prefix: str = ""):
        super(Describe, self).__init__()
        self.describe = lambda x: print(prefix + ndarray_helper.describe(x))

    def forward(self, x):
        _lh.debug(self.describe(x))
        return x
