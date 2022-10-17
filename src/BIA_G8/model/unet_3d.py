"""
From: https://github.com/aryaman4152/model-implementations-PyTorch
"""

__all__ = (
    "UNet3D",
)

import torch
from torch import nn
from torch.nn import functional as F


class Section3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super(Section3D, self).__init__()
        self.process = nn.Sequential(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding='same'
            ),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding='same'
            ),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        px = self.process(x)
        return px


class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet3D, self).__init__()
        # Contraction
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.down1 = Section3D(in_channels=in_channels, out_channels=64, kernel_size=3)
        self.down2 = Section3D(in_channels=64, out_channels=128, kernel_size=3)
        self.down3 = Section3D(in_channels=128, out_channels=256, kernel_size=3)
        self.down4 = Section3D(in_channels=256, out_channels=512, kernel_size=3)
        self.down5 = Section3D(in_channels=512, out_channels=1024, kernel_size=3)
        # Expansion
        self.up_conv1 = nn.ConvTranspose3d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        self.up_conv2 = nn.ConvTranspose3d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.up_conv3 = nn.ConvTranspose3d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.up_conv4 = nn.ConvTranspose3d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.up1 = Section3D(in_channels=1024, out_channels=512, kernel_size=3)
        self.up2 = Section3D(in_channels=512, out_channels=256, kernel_size=3)
        self.up3 = Section3D(in_channels=256, out_channels=128, kernel_size=3)
        self.up4 = Section3D(in_channels=128, out_channels=64, kernel_size=3)
        self.output = nn.Conv3d(
            in_channels=64,
            out_channels=out_channels,
            kernel_size=1,
            padding='same'
        )

    def forward(self, x):
        orig_x_size = x.shape
        skip_connections = []

        # CONTRACTION
        # down 1
        x = self.down1(x)
        skip_connections.append(x)
        x = self.pool(x)
        # down 2
        x = self.down2(x)
        skip_connections.append(x)
        x = self.pool(x)
        # down 3
        x = self.down3(x)
        skip_connections.append(x)
        x = self.pool(x)
        # down 4
        x = self.down4(x)
        skip_connections.append(x)
        x = self.pool(x)
        # down 5
        x = self.down5(x)

        # EXPANSION
        # up1
        x = self.up_conv1(x)
        y = skip_connections.pop()
        x = F.interpolate(x, y.shape[2:])
        y_new = torch.cat((y, x), dim=1)
        x = self.up1(y_new)
        # up2
        x = self.up_conv2(x)
        y = skip_connections.pop()
        # resize skip commention
        x = F.interpolate(x, y.shape[2:])
        y_new = torch.cat((y, x), dim=1)
        x = self.up2(y_new)
        # up3
        x = self.up_conv3(x)
        y = skip_connections.pop()
        x = F.interpolate(x, y.shape[2:])
        y_new = torch.cat((y, x), dim=1)
        x = self.up3(y_new)
        # up4
        x = self.up_conv4(x)
        y = skip_connections.pop()
        x = F.interpolate(x, y.shape[2:])
        y_new = torch.cat((y, x), dim=1)
        x = self.up4(y_new)

        x = self.output(x)
        x = F.interpolate(x, orig_x_size[2:])
        return x

# if __name__ == '__main__':
#     x = torch.randn((1, 1, 128, 128, 60))
#     pred = UNet3D(out_channels=3)(x)
#     print(pred.shape)
