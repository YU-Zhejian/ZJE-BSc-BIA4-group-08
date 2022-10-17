import glob
import os
import random
from typing import Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
import ray.data
import tqdm

from BIA_G8 import get_lh
from BIA_G8.helper import io_helper

_lh = get_lh(__name__)
_encoder = {
    "COVID-19": 0,
    "NORMAL": 1,
    "Viral_Pneumonia": 2
}

_decoder = {v: k for k, v in _encoder.items()}



def get_sklearn_dataset(
        dataset_path: str,
        desired_size=1024 * 1024,
        size: int = -1
) -> Tuple[npt.NDArray, npt.NDArray]:
    _lh.info("Requested data from %s...", dataset_path)
    all_image_paths = list(glob.glob(os.path.join(dataset_path, "*", "*.npy.xz")))
    _lh.info("Requested data from %s...found %d datasets", dataset_path, len(all_image_paths))
    if size != -1:
        all_image_paths = random.sample(all_image_paths, size)
    num_images = len(all_image_paths)
    X = np.ndarray((num_images, desired_size), dtype=float)
    y = np.ndarray((num_images,), dtype=int)
    for i, image_path in enumerate(tqdm.tqdm(
            iterable=all_image_paths,
            desc="loading data..."
    )):
        X[i] = np.ravel(io_helper.read_np_xz(image_path))
        y[i] = _encoder[os.path.split(os.path.split(image_path)[0])[1]]
    labels, counts = np.unique(y, return_counts=True)
    labels = [_decoder[label] for label in labels]
    _lh.info("Loaded labels %s with corresponding counts %s", str(labels), str(counts))

    return X, y

def get_ray_dataset(
        dataset_path: str,
        desired_size=1024 * 1024,
        size: int = -1
) -> ray.data.Dataset:
    _lh.info("Requested data from %s...", dataset_path)
    all_image_paths = list(glob.glob(os.path.join(dataset_path, "*", "*.npy.xz")))
    _lh.info("Requested data from %s...found %d datasets", dataset_path, len(all_image_paths))
    if size != -1:
        all_image_paths = random.sample(all_image_paths, size)
    X_df = pd.DataFrame(columns=list(map(str, range(desired_size))))
    y_df = pd.DataFrame(columns=["label"])
    for i, image_path in enumerate(tqdm.tqdm(
            iterable=all_image_paths,
            desc="loading data..."
    )):
        X_df = X_df.append(
            pd.DataFrame(np.ravel(io_helper.read_np_xz(image_path)))
        )
        y_df = y_df.append(
            pd.DataFrame(_encoder[os.path.split(os.path.split(image_path)[0])[1]])
        )
    df = pd.merge(X_df, y_df)

    return ray.data.from_pandas(df)

