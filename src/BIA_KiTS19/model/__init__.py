from torch import nn

from BIA_KiTS19.helper import ndarray_helper


class Describe(nn.Module):
    def __init__(self, prefix: str = ""):
        super(Describe, self).__init__()
        self.describe = lambda x: print(prefix + ndarray_helper.describe(x))

    def forward(self, x):
        self.describe(x)
        return x
