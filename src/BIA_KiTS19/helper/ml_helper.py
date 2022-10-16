__al__ = (
    "dice",
    "evaluate",
    "convert_2d_predictor_to_3d_predictor",
    "predict_2d",
    "predict_3d"
)

import functools
import operator
from typing import Callable, Union

import torch

from BIA_KiTS19 import get_lh
from BIA_KiTS19.helper import dataset_helper, ndarray_helper

_lh = get_lh(__name__)


def get_torch_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def dice(mask_1: torch.Tensor, mask_2: torch.Tensor) -> float:
    """Get similarity between 2 masks"""
    if mask_1.shape != mask_2.shape:
        raise ValueError()
    return torch.sum(mask_1 == mask_2) / functools.reduce(operator.mul, mask_1.shape)


def evaluate(
        predictor: Callable[[torch.Tensor], torch.Tensor],
        image_set: dataset_helper.ImageSet
) -> float:
    predicted_mask = predictor(image_set.tensor_image_final)
    retf = dice(predicted_mask, image_set.tensor_mask_final)
    _lh.debug(
        "Evaluate %s with predicted %s => %5.2f%%",
        ndarray_helper.describe(image_set.np_mask_final),
        ndarray_helper.describe(predicted_mask),
        retf * 100
    )
    return retf


def predict(net, image: torch.Tensor, device: Union[torch.device, str]) -> torch.Tensor:
    a = torch.argmax(
        torch.squeeze(
            net(
                torch.unsqueeze(
                    torch.unsqueeze(image, dim=0),
                    dim=0
                ).to(device)
            ),
            dim=0
        ),
        dim=0
    )
    return a


def convert_2d_predictor_to_3d_predictor(
        predictor_2d: Callable[[torch.Tensor], torch.Tensor],
        axis: int = 0
) -> Callable[[torch.Tensor], torch.Tensor]:
    def predictor(img: torch.Tensor) -> torch.Tensor:
        return ndarray_helper.merge_along_tensor(
            map(
                predictor_2d,
                ndarray_helper.sample_along_tensor(array=img, axis=axis)
            ),
            axis=axis
        )

    return predictor
