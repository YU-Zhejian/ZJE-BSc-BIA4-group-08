from __future__ import annotations

import torch

from BIA_G8.torch_modules import AbstractTorchModule


class ToyCNNModule(AbstractTorchModule):
    """
    A 2-layer small CNN that is very fast.

    Modified from <https://blog.csdn.net/qq_45588019/article/details/120935828>
    """

    def __init__(
            self,
            n_features: int,
            n_classes: int,
            kernel_size: int,
            stride: int,
            padding: int
    ):
        """
        :param n_features: Number of pixels in each image.
        :param n_classes: Number of output classes.
        :param kernel_size: Convolutional layer parameter.
        :param stride: Convolutional layer parameter.
        :param padding: Convolutional layer parameter.
        """
        super(ToyCNNModule, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            ),
            torch.nn.BatchNorm2d(num_features=16),
            torch.nn.ReLU()
        )
        """
        ``[BATCH_SIZE, 1, WID, HEIGHT] -> [BATCH_SIZE, 16, WID // 2, HEIGHT // 2]``
        """
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            ),
            torch.nn.BatchNorm2d(num_features=32),
            torch.nn.ReLU()
        )
        """
        ``[BATCH_SIZE, 16, WID // 2, HEIGHT // 2] -> [BATCH_SIZE, 32, WID // 4, HEIGHT // 4]``
        """
        self.mlp1 = torch.nn.Linear(
            in_features=32 * n_features // (4 ** 2),
            out_features=64
        )
        """
        ``[BATCH_SIZE, 32, WID // 4, HEIGHT // 4] -> [BATCH_SIZE, 64]``
        """
        self.mlp2 = torch.nn.Linear(
            in_features=64,
            out_features=n_classes
        )
        """
        ``[BATCH_SIZE, 64] -> [BATCH_SIZE, 3]``
        """
        # self.describe = Describe()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        # x = self.describe(x)
        x = self.conv1(x)
        # x = self.describe(x)
        x = self.conv2(x)
        # x = self.describe(x)
        x = self.mlp1(x.view(batch_size, -1))
        # x = self.describe(x)
        x = self.mlp2(x)
        # x = self.describe(x)
        return x


