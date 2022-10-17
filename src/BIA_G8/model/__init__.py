from torch import nn

from BIA_G8 import get_lh
from BIA_G8.helper import ndarray_helper

_lh = get_lh(__name__)


class Describe(nn.Module):
    def __init__(self, prefix: str = ""):
        super(Describe, self).__init__()
        self.describe = lambda x: print(prefix + ndarray_helper.describe(x))

    def forward(self, x):
        _lh.debug(self.describe(x))
        return x


class DiceLoss(nn.Module):
    # https://www.kaggle.com/code/yunkaili/brain-segmentation-pytorch

    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, y_pred, y_true):
        dsc = (2. * (y_pred * y_true).sum() + 1e-5) / (
                y_pred.sum() + y_true.sum() + 1e-5
        )
        return 1. - dsc
