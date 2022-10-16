import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

import BIA_KiTS19.helper.ml_helper
from BIA_KiTS19 import get_lh
from BIA_KiTS19.helper import dataset_helper, torch_helper, ml_helper
from BIA_KiTS19.model import unet_3d

device = "cpu"  # ml_helper.get_torch_device()

_lh = get_lh(__name__)

if __name__ == '__main__':
    train_dataset = dataset_helper.DataSet('/media/yuzj/BUP/kits19/data', range=(0, 5))
    _lh.info(f"{len(train_dataset)} train datasets loaded")
    test_dataset = dataset_helper.DataSet('/media/yuzj/BUP/kits19/data', range=(201, 209))
    _lh.info(f"{len(train_dataset)} test datasets loaded")
    test_case_names = list(test_dataset.iter_case_names())
    dataset = torch_helper.KiTS19DataSet(
        train_dataset
    )
    _lh.info(f"{len(dataset)} train images loaded")
    data_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True
    )
    net = unet_3d.UNET3D(
        in_channels=1,
        out_channels=3
    ).to(device)
    loss_fun = nn.CrossEntropyLoss()
    opt = optim.Adam(net.parameters())
    epoch = 0
    while True:
        for i, (image, mask) in enumerate(data_loader):
            image = image.to(device)
            mask = mask.to(device)
            # print("mask: " + ndarray_helper.describe(mask))
            # print("image: " + ndarray_helper.describe(image))
            predicted_mask = net(image)
            # print("predicted_mask: " + ndarray_helper.describe(np.array(predicted_mask.detach())))
            predicted_mask_argmax = torch.argmax(predicted_mask, dim=1)
            train_loss = loss_fun(predicted_mask, torch.squeeze(mask, dim=0))
            opt.zero_grad()
            train_loss.backward()
            opt.step()
            train_dice = np.mean(
                list(map(
                    lambda imageset: ml_helper.evaluate(
                        lambda img: BIA_KiTS19.helper.ml_helper.predict(net, img, device),
                        imageset
                    ),
                    test_dataset
                )))
            _lh.info('Epoch %d Image %d dice: %5.2f', epoch, i, train_dice * 100)
            if i % 50 == 0:
                _lh.info('%d figures processed', i)
        epoch += 1
