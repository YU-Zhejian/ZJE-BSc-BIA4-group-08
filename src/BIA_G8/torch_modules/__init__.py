"""
Self-created torch modules.
"""

from typing import Callable

import torch
from torch import nn


class AbstractTorchModule(nn.Module):
    """:py:class:`nn.Module` with better type hints"""

    def __init__(self, **params) -> None:
        super().__init__()

    __call__: Callable[[torch.Tensor], torch.Tensor]
