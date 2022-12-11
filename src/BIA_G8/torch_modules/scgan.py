"""
The SCGAN module for super resolution.

Modified from <https://blog.csdn.net/qianbin3200896/article/details/104181552>
"""

import math
from typing import Optional

import torch
import torchvision
from torch import nn

from BIA_G8.torch_modules import AbstractTorchModule


class ConvolutionalBlock(AbstractTorchModule):
    """
    Convolution model is consisted of layers of Convolution, Batch Normalization, and Activation
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int,
            include_batch_norm: bool,
            activation: Optional[str]
    ):
        """
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param kernel_size: Convolutional kernel size.
        :param stride: Convolutional stride size.
        :param include_batch_norm: Whether to include a batch normalization layr.
        :param activation: Whether to include an activation layer and type if True.
            Should be one of ``('prelu', 'leakyrelu', 'tanh')``.
        """
        super().__init__()

        if activation is not None:
            activation = activation.lower()
            assert activation in ('prelu', 'leakyrelu', 'tanh')

        layers = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2
            )
        ]

        if include_batch_norm:
            layers.append(nn.BatchNorm2d(num_features=out_channels))

        if activation == 'prelu':
            layers.append(nn.PReLU())
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU(0.2))
        elif activation == 'tanh':
            layers.append(nn.Tanh())

        self.conv_block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_block(x)


class SRResNetSubPixelConvolutionalBlock(AbstractTorchModule):
    """
    Sub-pixel convolution module, including convolution, pixel cleaning and activation layers
    """

    def __init__(
            self,
            kernel_size: int,
            n_channels: int,
            scale_factor: int
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=n_channels,
            out_channels=n_channels * (scale_factor ** 2),
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        """
         ``[BATCH_SIZE, N_CHANNELS, WID, HEIGHT] -> [BATCH_SIZE, N_CHANNELS * SF ^ 2, WID, HEIGHT]``
        """

        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=scale_factor)
        """
         ``[BATCH_SIZE, N_CHANNELS * SF ^ 2, WID, HEIGHT] -> [BATCH_SIZE, N_CHANNELS, WID * SF, HEIGHT * SF]``
        """
        self.prelu = nn.PReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.conv(x)
        output = self.pixel_shuffle(output)
        output = self.prelu(output)

        return output


class SRResNetResidualBlock(AbstractTorchModule):
    """
    The residual module consists of two convolution modules and a skip connection
    """

    def __init__(
            self,
            kernel_size: int,
            n_channels: int
    ):
        super().__init__()

        self.conv_block1 = ConvolutionalBlock(
            in_channels=n_channels,
            out_channels=n_channels,
            kernel_size=kernel_size,
            include_batch_norm=True,
            stride=1,
            activation='PReLu'
        )
        """
         ``[BATCH_SIZE, N_CHANNELS, WID, HEIGHT] -> [BATCH_SIZE, N_CHANNELS * SF ^ 2, WID, HEIGHT]``
        """
        self.conv_block2 = ConvolutionalBlock(
            in_channels=n_channels,
            out_channels=n_channels,
            kernel_size=kernel_size,
            include_batch_norm=True,
            stride=1,
            activation=None
        )
        """
         ``[BATCH_SIZE, N_CHANNELS, WID, HEIGHT] -> [BATCH_SIZE, N_CHANNELS * SF ^ 2, WID, HEIGHT]``
        """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        output = self.conv_block1(x)
        output = self.conv_block2(output)
        output = output + residual
        return output


class SRResNet(AbstractTorchModule):
    """
    The module of SRResNET
    """

    def __init__(
            self,
            large_kernel_size: int,
            small_kernel_size: int,
            in_channels: int,
            n_intermediate_channels: int,
            n_blocks: int,
            scale_factor: int
    ):
        """
        :param large_kernel_size: Kernel size of the first and last layer
        :param small_kernel_size: Kernel size of the intermediate layers
        :param n_intermediate_channels: Number of channels of the intermediate layers
        :param n_blocks: Number of residual_blocks
        :param scale_factor: Scaling factor, should be one of 2, 4, 8
        """
        super(SRResNet, self).__init__()

        scale_factor = int(scale_factor)
        assert scale_factor in (2, 4, 8), "Incorrect scaling factor -- "

        self.conv_block1 = ConvolutionalBlock(
            in_channels=in_channels,
            out_channels=n_intermediate_channels,
            kernel_size=large_kernel_size,
            include_batch_norm=False,
            stride=1,
            activation='PReLu'
        )
        """
        ``[BATCH_SIZE, N_INTERMEDIATE_CHANNELS, WID, HEIGHT] -> [BATCH_SIZE, N_INTERMEDIATE_CHANNELS, WID, HEIGHT]``
        """

        self.residual_blocks = nn.Sequential(
            *([SRResNetResidualBlock(kernel_size=small_kernel_size, n_channels=n_intermediate_channels)] * n_blocks)
        )
        """
        ``[BATCH_SIZE, N_INTERMEDIATE_CHANNELS, WID, HEIGHT] -> [BATCH_SIZE, N_INTERMEDIATE_CHANNELS, WID, HEIGHT]``
        """

        self.conv_block2 = ConvolutionalBlock(
            in_channels=n_intermediate_channels,
            out_channels=n_intermediate_channels,
            kernel_size=small_kernel_size,
            include_batch_norm=True,
            stride=1,
            activation=None
        )
        """
        ``[BATCH_SIZE, N_INTERMEDIATE_CHANNELS, WID, HEIGHT] -> [BATCH_SIZE, N_INTERMEDIATE_CHANNELS, WID, HEIGHT]``
        """

        n_subpixel_convolution_blocks = int(math.log2(scale_factor))
        self.subpixel_convolutional_blocks = nn.Sequential(
            *([SRResNetSubPixelConvolutionalBlock(
                kernel_size=small_kernel_size,
                n_channels=n_intermediate_channels,
                scale_factor=2
            )] * n_subpixel_convolution_blocks)
        )
        """
        ``[BATCH_SIZE, N_INTERMEDIATE_CHANNELS, WID, HEIGHT] ->
         [BATCH_SIZE, N_INTERMEDIATE_CHANNELS, WID * SF, HEIGHT * SF]``
        """

        self.conv_block3 = ConvolutionalBlock(
            in_channels=n_intermediate_channels,
            out_channels=3,
            kernel_size=large_kernel_size,
            include_batch_norm=False,
            stride=1,
            activation='Tanh'
        )
        """
        ``[BATCH_SIZE, N_INTERMEDIATE_CHANNELS, WID * SF, HEIGHT * SF] ->
         [BATCH_SIZE, N_CHANNELS, WID * SF, HEIGHT * SF]``
        """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.conv_block1(x)
        residual = output
        output = self.residual_blocks(output)
        output = self.conv_block2(output)
        output = output + residual
        output = self.subpixel_convolutional_blocks(output)
        sr_imgs = self.conv_block3(output)
        return sr_imgs


class SCGANGenerator(nn.Module):
    """
    The module of the generator, the structure of which is almost the same as module SRResNET
    """

    def __init__(
            self,
            large_kernel_size: int,
            small_kernel_size: int,
            in_channels: int,
            n_intermediate_channels: int,
            n_blocks: int,
            scale_factor: int
    ):
        super().__init__()
        self.net = SRResNet(
            large_kernel_size=large_kernel_size,
            small_kernel_size=small_kernel_size,
            in_channels=in_channels,
            n_intermediate_channels=n_intermediate_channels, n_blocks=n_blocks,
            scale_factor=scale_factor
        )
        """
        ``[BATCH_SIZE, N_CHANNELS, WID, HEIGHT] -> [BATCH_SIZE, N_CHANNELS, WID * SF, HEIGHT * SF]``
        """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SCGANDiscriminator(nn.Module):
    """
    SRGAN discriminator
    """

    def __init__(
            self,
            kernel_size: int,
            n_channels: int,
            in_channels: int,
            n_blocks: int,
            fc_size: int
    ):
        """
        :param kernel_size: Number of kernel size of all convolutional layers
        :param n_channels: Initial number of channels in convolutional layer
        :param n_blocks: Number of convolutional blocks
        :param fc_size: Number of fully connected layers
        """
        super(SCGANDiscriminator, self).__init__()
        out_channels = in_channels

        conv_blocks = []
        for i in range(n_blocks):
            out_channels = (n_channels if i == 0 else in_channels * 2) if i % 2 == 0 else in_channels
            conv_blocks.append(
                ConvolutionalBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=1 if i % 2 == 0 else 2,
                    include_batch_norm=i != 0,
                    activation='LeakyReLu'
                )
            )
            in_channels = out_channels
        self.conv_blocks = nn.Sequential(*conv_blocks)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((6, 6))
        self.fc1 = nn.Linear(out_channels * 6 * 6, fc_size)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(1024, 1)

        # BCEWithLogitsLoss incldues sigmoid layer, so not needed.

    def forward(self, imgs):
        """
        ``[BATCH_SIZE, N_INTERMEDIATE_CHANNELS, WID * SF, HEIGHT * SF] ->
        [BATCH_SIZE]``
        """
        batch_size = imgs.size(0)
        output = self.conv_blocks(imgs)
        output = self.adaptive_pool(output)
        output = self.fc1(output.view(batch_size, -1))
        output = self.leaky_relu(output)
        logit = self.fc2(output)

        return logit


class TruncatedVGG19(nn.Module):
    """
    The network of truncated VGG19 that is used to calculate the MSE loss of VGG feature space
    """

    def __init__(self, i, j):
        """
        :param i: Truncate at ith max pooling layer
        :param j: Truncate at jth convolutional layer
        """
        super().__init__()

        vgg19 = torchvision.models.vgg19()
        maxpool_counter = 0
        conv_counter = 0
        truncate_at = 0
        for layer in vgg19.features.children():
            truncate_at += 1

            if isinstance(layer, nn.Conv2d):
                conv_counter += 1
            if isinstance(layer, nn.MaxPool2d):
                maxpool_counter += 1
                conv_counter = 0
            if maxpool_counter == i - 1 and conv_counter == j:
                break
        else:
            raise TypeError

        self.truncated_vgg19 = nn.Sequential(*list(vgg19.features.children())[:truncate_at + 1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.truncated_vgg19(x)
        return output
