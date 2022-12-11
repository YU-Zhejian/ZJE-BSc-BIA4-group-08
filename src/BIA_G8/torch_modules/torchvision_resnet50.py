from __future__ import annotations

from torchvision.models import resnet50

from BIA_G8.torch_modules import AbstractTorchModule


class TorchVisionResnet50Module(AbstractTorchModule):
    """
    Wrapper to resnet50 provided by :py:mod:`torchvision`.
    """

    def __new__(cls, *args, **kwargs):
        return resnet50()
