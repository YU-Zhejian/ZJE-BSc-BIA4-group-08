from __future__ import annotations

import copy
import glob
import os
import random
from collections import defaultdict
from typing import Tuple, List, Dict, Callable, Iterable, Optional

import numpy as np
import numpy.typing as npt
import pandas as pd
import ray.data
import tqdm

import skimage.io as skiio

from BIA_G8 import get_lh
from BIA_G8.helper import io_helper, joblib_helper

_lh = get_lh(__name__)
_encoder = {
    "COVID-19": 0,
    "NORMAL": 1,
    "Viral_Pneumonia": 2
}

_decoder = {v: k for k, v in _encoder.items()}

IN_MEMORY_INDICATOR = "IN_MEMORY"


def resolve_label_from_path(abspath: str) -> str:
    return os.path.split(os.path.split(abspath)[0])[1]


class CovidImage:
    _label: int
    _label_str: str
    _image_path: str
    _image: npt.NDArray

    @classmethod
    def from_file(cls, image_path: str):
        new_img = cls()
        new_img._image_path = image_path
        new_img._label_str = resolve_label_from_path(new_img._image_path)
        new_img._label = _encoder[new_img.label_str]

        if image_path.endswith("npy.xz"):
            new_img._image = io_helper.read_np_xz(image_path)
        else:
            new_img._image = skiio.imread(image_path)
        return new_img

    @property
    def label(self) -> int:
        return self._label

    @property
    def label_str(self) -> str:
        return self._label_str

    @property
    def as_np_array(self) -> npt.NDArray:
        return self._image

    @classmethod
    def from_image(cls, image: npt.NDArray, label: int):
        new_img = cls()
        new_img._image_path = IN_MEMORY_INDICATOR
        new_img._image = image
        new_img._label = label
        new_img._label_str = _decoder[new_img._label]
        return new_img

    def apply(self, operation: Callable[[npt.NDArray], npt.NDArray]) -> CovidImage:
        return CovidImage.from_image(
            operation(self._image),
            self._label
        )

    def save(self, image_path: Optional[str] = None):
        if image_path is None:
            image_path = self._image_path
        self._image_path = image_path
        if image_path.endswith(".npy.xz"):
            io_helper.write_np_xz(self._image, image_path)
        else:
            skiio.imsave(image_path, self._image)


class CovidDataSet():
    _all_image_paths: List[str]
    _image_paths_with_label: Dict[str, List[str]]
    _loaded_image: List[CovidImage]
    _dataset_path: str

    def __init__(
            self,
            dataset_path: str
    ):
        self._dataset_path = dataset_path
        if self._dataset_path == IN_MEMORY_INDICATOR:
            _lh.info("Requested in-memory dataset")
            self._all_image_paths = []
            self._image_paths_with_label = {}
            return
        _lh.info("Requested data from %s...", self._dataset_path)
        _all_image_paths = list(glob.glob(os.path.join(self._dataset_path, "*", "*.npy.xz")))
        self._image_paths_with_label = defaultdict(lambda: [])
        for image_path in tqdm.tqdm(
                iterable=_all_image_paths,
                desc="Parsing image directory..."
        ):
            label = resolve_label_from_path(image_path)
            self._image_paths_with_label[label].append(image_path)
            self._all_image_paths.append(image_path)

    @classmethod
    def from_loaded_image(cls, loaded_image: List[CovidImage]):
        new_ds = cls(IN_MEMORY_INDICATOR)
        new_ds._loaded_image = loaded_image
        return new_ds

    def load(self, size: int = -1, balanced: bool = True):
        if self._dataset_path == IN_MEMORY_INDICATOR:
            raise ValueError("Cannot reload an in-memory dataset")
        self._loaded_image = []
        max_balanced_size = min(map(len, self._image_paths_with_label.values())) * 3
        if size == -1:
            if balanced:
                size = max_balanced_size
            else:
                size = len(self._all_image_paths)
        if balanced:
            if size > max_balanced_size:
                raise ValueError("Requested too many images!")
        else:
            if size > len(self._all_image_paths):
                raise ValueError("Requested too many images!")
        _lh.info("Loading %d data from %s...", size, self._dataset_path)
        if balanced:
            for image_path in random.sample(self._all_image_paths, size):
                self._loaded_image.append(CovidImage.from_file(image_path))
        else:
            for image_paths in self._image_paths_with_label.values():
                for image_path in random.sample(image_paths, size // 3):
                    self._loaded_image.append(CovidImage.from_file(image_path))
        _lh.info("Shuffling loaded data...")
        random.shuffle(self._loaded_image)
        _lh.info("Finished loading data...")

    def __len__(self):
        return len(self._loaded_image)

    def __iter__(self) -> Iterable[CovidImage]:
        return iter(self._loaded_image)

    def __getitem__(self, i: int) -> CovidImage:
        return self._loaded_image[i]

    def __setitem__(self, i: int, value: CovidImage):
        self._loaded_image[i] = value

    def apply(
            self,
            operation: Callable[[npt.NDArray], npt.NDArray],
            n_jobs:int
    ) -> CovidDataSet:
        new_ds = CovidDataSet.from_loaded_image(list(joblib_helper.parallel_map(
            lambda image: image.apply(operation),
            self._loaded_image,
            n_jobs=n_jobs
        )))
        return new_ds


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
        X_df = pd.concat([
            X_df,
            pd.DataFrame(np.ravel(io_helper.read_np_xz(image_path)).reshape((1, desired_size)))
        ])
        y_df = pd.concat([
            y_df,
            pd.DataFrame([_encoder[os.path.split(os.path.split(image_path)[0])[1]]])
        ])
    df = pd.merge(X_df, y_df)

    return ray.data.from_pandas(df)
