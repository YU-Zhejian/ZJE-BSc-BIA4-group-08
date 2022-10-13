import random

import numpy as np
import numpy.typing as npt
import torch
import torchvision.transforms.functional as TF
from torch import nn, optim
from torch.utils.data import DataLoader

from BIA_KiTS19.helper import dataset_helper, torch_helper, converter, ml_helper, ndarray_helper

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Section(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Section, self).__init__()
        self.process = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding='same'
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding='same'
            ),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.process(x)


class UNET(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNET, self).__init__()
        # Contraction
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down1 = Section(in_channels=in_channels, out_channels=64, kernel_size=3)
        self.down2 = Section(in_channels=64, out_channels=128, kernel_size=3)
        self.down3 = Section(in_channels=128, out_channels=256, kernel_size=3)
        self.down4 = Section(in_channels=256, out_channels=512, kernel_size=3)
        self.down5 = Section(in_channels=512, out_channels=1024, kernel_size=3)
        # Expansion
        self.up_conv1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        self.up_conv2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.up_conv3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.up_conv4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.up1 = Section(in_channels=1024, out_channels=512, kernel_size=3)
        self.up2 = Section(in_channels=512, out_channels=256, kernel_size=3)
        self.up3 = Section(in_channels=256, out_channels=128, kernel_size=3)
        self.up4 = Section(in_channels=128, out_channels=64, kernel_size=3)
        self.output = self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1, padding='same')

    def forward(self, x):
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
        y = skip_connections[3]
        x = TF.resize(x, y.shape[2:])
        y_new = torch.cat((y, x), dim=1)
        x = self.up1(y_new)
        # up2
        x = self.up_conv2(x)
        y = skip_connections[2]
        # resize skip commention
        x = TF.resize(x, y.shape[2:])
        y_new = torch.cat((y, x), dim=1)
        x = self.up2(y_new)
        # up3
        x = self.up_conv3(x)
        y = skip_connections[1]
        x = TF.resize(x, y.shape[2:])
        y_new = torch.cat((y, x), dim=1)
        x = self.up3(y_new)
        # up4
        x = self.up_conv4(x)
        y = skip_connections[0]
        x = TF.resize(x, y.shape[2:])
        y_new = torch.cat((y, x), dim=1)
        x = self.up4(y_new)

        x = self.output(x)
        return x


@np.vectorize
def reshape(x:float) -> float:
    if x < 1/3:
        return 0
    elif x < 2/3:
        return 0.5
    else:
        return 1

def predict_2d(net, image: npt.NDArray[float]) -> npt.NDArray[float]:
    return converter.tensor_to_np_2d(
        net(torch.unsqueeze(converter.np_2d_to_tensor(image), dim=0))
    )


if __name__ == '__main__':
    train_dataset = dataset_helper.DataSet('/media/yuzj/BUP/kits19/data', range=(0, 3))
    test_dataset = dataset_helper.DataSet('/media/yuzj/BUP/kits19/data', range=(21, 25))
    test_case_names = list(test_dataset.iter_case_names())
    data_loader = DataLoader(
        torch_helper.KiTS19DataSet2D(
            train_dataset,
            axis=2
        ),
        batch_size=100,
        shuffle=True
    )
    net = UNET().to(device)
    loss_fun = nn.MSELoss()
    opt = optim.Adam(net.parameters())
    epoch = 0
    while True:
        for i, (image, mask) in enumerate(data_loader):
            image, mask = image.to(device), mask.to(device)

            predicted_mask = net(image)
            train_loss = loss_fun(predicted_mask, mask)

            opt.zero_grad()
            train_loss.backward()
            opt.step()

            if i % 5 == 0:
                train_dice = ml_helper.evaluate(
                    ml_helper.convert_2d_predictor_to_3d_predictor(
                        lambda img: predict_2d(net, img),
                        axis=2
                    ),
                    test_dataset[random.choice(test_case_names)]
                )
                print(f'{epoch}-{i}-train_loss===>>{train_loss.item()} dice: {train_dice}')
        epoch += 1
