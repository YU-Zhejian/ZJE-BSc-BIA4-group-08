import statistics
from typing import Any, Callable, Tuple, Union

import numpy as np
import torch
import torch.utils.data as tud

from BIA_G8 import dataset_helper, ml_helper
from BIA_G8 import get_lh
from BIA_G8.helper import ndarray_helper

_lh = get_lh(__name__)


def evaluate_over_test_dataset(
        predictor: Callable[[torch.Tensor], torch.Tensor],
        test_dataset: dataset_helper.KiTS19DataSet
) -> float:
    return statistics.mean(
        map(
            lambda imageset: ml_helper.evaluate(
                predictor,
                imageset
            ),
            test_dataset
        ))


def mask_one_hot_encoder(mask: torch.Tensor) -> torch.Tensor:
    mask = torch.squeeze(mask, dim=1)
    print(mask.shape)
    colors = [0, 1, 2]
    mask_4d = torch.zeros(size=(len(colors), *mask.shape)).int()
    print(mask_4d.shape)

    for i in range(len(colors)):
        mask_4d[i, ...] = 1 * (mask == colors[i])
    mask_4d = torch.moveaxis(mask_4d, 0, 1)
    return mask_4d


def exec_epoch(
        train_torch_dataset: tud.Dataset,
        test_dataset: dataset_helper.KiTS19DataSet,
        net: Any,
        loss_fun: Any,
        opt: Any,
        predictor: Callable[[torch.Tensor], torch.Tensor],
        device: Union[torch.device, str]
) -> Tuple[float, float]:
    train_data_loader = tud.DataLoader(
        train_torch_dataset,
        batch_size=10,
        shuffle=True
    )
    accumulated_train_loss = []
    for i, (image, mask) in enumerate(train_data_loader):
        _lh.debug("Batch %d IMAGE: %s", i, ndarray_helper.describe(image))
        image = image.to(device)
        _lh.debug("Batch %d MASK: %s", i, ndarray_helper.describe(mask))
        mask_4d = mask_one_hot_encoder(mask)
        _lh.debug("Batch %d MASK [4D]: %s", i, ndarray_helper.describe(mask))
        mask_4d = mask_4d.to(device)
        predicted_mask = net(image)
        predicted_mask_argmax = torch.argmax(predicted_mask, dim=1)
        _lh.debug("Batch %d PMASK: %s", i, ndarray_helper.describe(np.array(predicted_mask.cpu().detach())))
        _lh.debug("Batch %d PMASK [ARGMAX]: %s", i,
                  ndarray_helper.describe(np.array(predicted_mask_argmax.cpu().detach())))
        train_loss = loss_fun(predicted_mask, mask_4d)
        train_loss.requires_grad_(True)
        accumulated_train_loss.append(train_loss.item())
        opt.zero_grad()
        train_loss.backward()
        opt.step()
        if i % 50 == 0:
            _lh.info('%d images processed', i)
    return evaluate_over_test_dataset(predictor, test_dataset), statistics.mean(accumulated_train_loss)
