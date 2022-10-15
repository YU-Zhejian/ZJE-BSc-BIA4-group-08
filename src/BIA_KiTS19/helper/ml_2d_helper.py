from typing import Callable

import numpy as np
import torch
from numpy import typing as npt

from BIA_KiTS19.helper import converter, ndarray_helper


def predict_2d(net, image: npt.NDArray[float], device: torch.device) -> npt.NDArray[float]:
    a = converter.tensor_to_np_2d(
        torch.squeeze(
            net(
                torch.unsqueeze(converter.np_to_tensor_2d(image), dim=0).to(device)
            ),
            dim=0
        )
    )
    return a


def predict_3d(net, image: npt.NDArray[float], device: torch.device) -> npt.NDArray[float]:
    a = np.array(
        torch.squeeze(
            net(
                torch.unsqueeze(torch.tensor(image), dim=0).to(device)
            ),
            dim=0
        )
    )
    return a


def convert_2d_predictor_to_3d_predictor(
        predictor_2d: Callable[[npt.NDArray[float]], npt.NDArray[float]],
        axis: int = 0
) -> Callable[[npt.NDArray[float]], npt.NDArray[float]]:
    def predictor(img: npt.NDArray[float]) -> npt.NDArray[float]:
        return np.argmax(
            ndarray_helper.merge_along_np(
                map(
                    predictor_2d,
                    ndarray_helper.sample_along_np(array=img, axis=axis)
                ),
                axis=axis
            ),
            axis=3
        )

    return predictor
