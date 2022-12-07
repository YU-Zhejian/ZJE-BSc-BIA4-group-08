"""
Self-created torch modules.
"""
from abc import abstractmethod
from typing import Callable

import torch
from torch import nn


class AbstractTorchModule(nn.Module):
    """:py:class:`nn.Module` with better type hints"""

    def __init__(self, **params) -> None:
        super().__init__()

    __call__: Callable[[torch.Tensor], torch.Tensor]

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
