from __future__ import annotations

__all__ = (
    "encode",
    "decode",
    "IN_MEMORY_INDICATOR",
    "CovidImage",
    "CovidDataSet",
    "resolve_label_from_path"
)

import glob
import itertools
import os
import random
import uuid
from collections import defaultdict
from typing import Tuple, List, Dict, Callable, Iterable, Optional, Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import ray.data
import skimage.io as skiio
import tqdm

from BIA_G8 import get_lh
from BIA_G8.helper import io_helper, joblib_helper

_lh = get_lh(__name__)
_encoder = {
    "NA": 100,
    "COVID-19": 0,
    "NORMAL": 1,
    "Viral_Pneumonia": 2
}

_decoder = {v: k for k, v in _encoder.items()}

IN_MEMORY_INDICATOR = "IN_MEMORY"


def encode(label_str: str) -> int:
    return _encoder[label_str]


def decode(label: int) -> str:
    return _decoder[label]


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

    @property
    def image_path(self):
        return self._image_path

    def save(self, image_path: Optional[str] = None):
        if image_path is None:
            image_path = self._image_path
        if image_path == IN_MEMORY_INDICATOR:
            raise ValueError("Cannot save to in-memory data.")
        self._image_path = image_path
        if image_path.endswith(".npy.xz"):
            io_helper.write_np_xz(self._image, image_path)
        else:
            skiio.imsave(image_path, self._image)


def _get_max_size_helper(
        size: int,
        balanced: bool,
        full_size: int,
        image_with_label: Dict[Any, Any]
):
    if balanced:
        max_size = min(map(len, image_with_label.values())) * 3
    else:
        max_size = full_size
    if size == -1:
        size = max_size
    if size > max_size:
        raise ValueError(f"Requested too many images! Max: {max_size}")
    return size


class CovidDataSet:
    _loaded_image: List[CovidImage]
    _loaded_image_with_label: Dict[int, List[CovidImage]]
    _dataset_path: str

    @property
    def dataset_path(self):
        return self._dataset_path

    def __init__(self):
        self._loaded_image = []
        self._loaded_image_with_label = defaultdict(lambda: [])
        self._dataset_path = IN_MEMORY_INDICATOR

    @classmethod
    def from_directory(
            cls,
            dataset_path: str,
            size: int = -1,
            balanced: bool = True
    ):
        new_ds = cls()
        new_ds._dataset_path = dataset_path
        _lh.info("Requested data from %s...", new_ds._dataset_path)
        _all_image_paths = list(glob.glob(os.path.join(new_ds._dataset_path, "*", "*.npy.xz")))
        _image_paths_with_label = defaultdict(lambda: [])
        for image_path in tqdm.tqdm(
                iterable=_all_image_paths,
                desc="Parsing image directory..."
        ):
            label = encode(resolve_label_from_path(image_path))
            _image_paths_with_label[label].append(image_path)
        size = _get_max_size_helper(size, balanced, len(_all_image_paths), _image_paths_with_label)
        _lh.info("Loading %d data from %s...", size, new_ds._dataset_path)

        if balanced:
            _all_image_paths = list(itertools.chain(
                *list(map(
                    lambda image_paths: random.sample(image_paths, size // 3),
                    _image_paths_with_label.values()
                ))
            ))
        for image_path in tqdm.tqdm(
                random.sample(_all_image_paths, size),
                desc="Loading data..."
        ):
            img = CovidImage.from_file(image_path)
            new_ds._loaded_image.append(img)
            new_ds._loaded_image_with_label[img.label].append(img)
        _lh.info("Shuffling loaded data...")
        random.shuffle(new_ds._loaded_image)
        _lh.info("Finished loading data...")
        return new_ds

    @classmethod
    def from_loaded_image(cls, loaded_image: List[CovidImage]):
        new_ds = cls()
        new_ds._loaded_image = loaded_image
        for img in new_ds._loaded_image:
            new_ds._loaded_image_with_label[img.label].append(img)
        return new_ds

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
            operation: Callable[[npt.NDArray], npt.NDArray]
    ) -> CovidDataSet:
        new_ds = CovidDataSet.from_loaded_image(list(map(
            lambda image: image.apply(operation),
            self._loaded_image
        )))
        return new_ds

    def parallel_apply(
            self,
            operation: Callable[[npt.NDArray], npt.NDArray],
            **kwargs
    ) -> CovidDataSet:
        new_ds = CovidDataSet.from_loaded_image(list(joblib_helper.parallel_map(
            lambda image: image.apply(operation),
            self._loaded_image,
            **kwargs
        )))
        return new_ds

    def sample(self, size: int = -1, balanced: bool = True):
        new_ds = CovidDataSet()
        new_ds._dataset_path = self.dataset_path
        _lh.info("Sampling data...")
        size = _get_max_size_helper(size, balanced, len(self._loaded_image), self._loaded_image_with_label)
        _lh.info("Loading %d data from %s...", size, new_ds._dataset_path)
        if balanced:
            for image_catagory in self._loaded_image_with_label.values():
                for img in random.sample(image_catagory, size // 3):
                    new_ds._loaded_image.append(img)
                    new_ds._loaded_image_with_label[img.label].append(img)
        else:
            for img in random.sample(self._loaded_image, size):
                new_ds._loaded_image.append(img)
                new_ds._loaded_image_with_label[img.label].append(img)
        _lh.info("Shuffling loaded data...")
        random.shuffle(new_ds._loaded_image)
        _lh.info("Finished loading data...")
        return new_ds

    def _save_impl(self, img: CovidImage, dataset_path: str):
        if img.image_path == IN_MEMORY_INDICATOR:
            _image_path = str(uuid.uuid4()) + ".npy.xz"
        else:
            _image_path = img.image_path
        img.save(os.path.join(dataset_path, img.label_str, os.path.basename(_image_path)))

    def _presave_hook(self, dataset_path: str):
        if dataset_path is None:
            dataset_path = self._dataset_path
        if dataset_path == IN_MEMORY_INDICATOR:
            raise ValueError("Cannot save to in-memory data.")
        self._dataset_path = dataset_path
        os.makedirs(self._dataset_path, exist_ok=True)
        for label in self._loaded_image_with_label.keys():
            label_str = decode(label)
            os.makedirs(os.path.join(self._dataset_path, label_str), exist_ok=True)

    def save(self, dataset_path: str = None):
        self._presave_hook(dataset_path)
        _ = list(map(
            lambda img: self._save_impl(img, dataset_path),
            tqdm.tqdm(
                iterable=list(itertools.chain(*self._loaded_image_with_label.values())),
                desc="Saving images..."
            )
        ))

    def parallel_save(self, dataset_path: str = None, **kwargs):
        self._presave_hook(dataset_path)
        _ = list(joblib_helper.parallel_map(
            lambda img: self._save_impl(img, dataset_path),
            tqdm.tqdm(
                iterable=list(itertools.chain(*self._loaded_image_with_label.values())),
                desc="Saving images..."
            ),
            **kwargs
        ))


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
