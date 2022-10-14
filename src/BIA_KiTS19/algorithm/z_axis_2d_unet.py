import random

import numpy.typing as npt
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from BIA_KiTS19 import get_lh
from BIA_KiTS19.helper import dataset_helper, torch_helper, converter, ml_helper
from BIA_KiTS19.model import unet_2d

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = "cpu"

_lh = get_lh(__name__)


def predict_2d(net, image: npt.NDArray[float]) -> npt.NDArray[float]:
    return converter.tensor_to_np(torch.squeeze(net(torch.unsqueeze(torch.Tensor(image), dim=0)), dim=0))


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
    net = unet_2d.UNET(
        in_channels=1,
        out_channels=3
    ).to(device)
    loss_fun = nn.CrossEntropyLoss()
    opt = optim.Adam(net.parameters())
    epoch = 0
    while True:
        for i, (image, mask) in enumerate(data_loader):
            image, mask = image.to(device), mask.to(device)

            predicted_mask = torch.argmax(net(image))
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
