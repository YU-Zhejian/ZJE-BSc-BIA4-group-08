"""
Helper classes and functions to general-purposed machine-learning/deep-learning over COVID image dataset.
"""

from __future__ import annotations

import doctest
import shutil
from functools import reduce

__all__ = (
    "encode",
    "decode",
    "IN_MEMORY_INDICATOR",
    "VALID_IMAGE_EXTENSIONS",
    "CovidImage",
    "CovidDataSet",
    "resolve_label_from_path"
)

import glob
import itertools
import operator
import os
import random
import uuid
from collections import defaultdict
from typing import Tuple, List, Dict, Callable, Iterable, Optional, Any

import numpy as np
import numpy.typing as npt
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
VALID_IMAGE_EXTENSIONS = (
    "npy.xz",
    "png",
    "jpg",
    "jpeg",
    "tif",
    "tiff"
)
"""Image extensions that can be parsed by the reader."""

IN_MEMORY_INDICATOR = "IN_MEMORY"
"""\"File name\" of the in-memory datasets or figures."""


def encode(label_str: str) -> int:
    """
    Categorical encoder.

    :param label_str: Label in string format.
    :return: Label in integer format.
    """
    return _encoder[label_str]


def decode(label: int) -> str:
    """
    Categorical decoder.

    :param label: Label in integer format.
    :return: Label in string format.
    """
    return _decoder[label]


def resolve_label_from_path(abspath: str) -> str:
    """
    Get image paths from label name.

    >>> resolve_label_from_path("/100/200/300.png")
    '200'

    :param abspath: Absolute path to the image.
    :return: Resolved label in string format.
    """
    return os.path.split(os.path.split(abspath)[0])[1]


def _get_max_size_helper(
        size: int,
        balanced: bool,
        full_size: int,
        image_with_label: Dict[Any, Any]
) -> int:
    if balanced:
        max_size = min(map(len, image_with_label.values())) * 3
    else:
        max_size = full_size
    if size == -1:
        size = max_size
    if size > max_size:
        raise ValueError(f"Requested too many images! Max: {max_size}")
    return size


class CovidImage:
    """
    The COVID image infrastructure.
    """
    _label: int
    _label_str: str
    _image_path: str
    _image: npt.NDArray

    @property
    def label(self) -> int:
        """Read-only label integer."""
        return self._label

    @property
    def label_str(self) -> str:
        """Read-only label string."""
        return self._label_str

    @property
    def np_array(self) -> npt.NDArray:
        """Read-only numpy aray of the image."""
        return self._image

    @property
    def image_path(self) -> str:
        """Read-only absolute path to the image."""
        return self._image_path

    @classmethod
    def from_file(cls, image_path: str) -> CovidImage:
        """
        Generate a new instance from existing file.

        :param image_path: Absolute path to the image.
        """
        new_img = cls()
        new_img._image_path = image_path
        new_img._label_str = resolve_label_from_path(new_img._image_path)
        new_img._label = encode(new_img.label_str)

        if image_path.endswith("npy.xz"):
            new_img._image = io_helper.read_np_xz(image_path)
        else:
            new_img._image = skiio.imread(image_path)
        return new_img

    @classmethod
    def from_np_array(cls, image: npt.NDArray, label: int) -> CovidImage:
        """
        Generate a new instance based on existing Numpy array.

        :param image: Image in numpy aray.
        :param label: The label in integer.
        """
        new_img = cls()
        new_img._image_path = IN_MEMORY_INDICATOR
        new_img._image = image
        new_img._label = label
        new_img._label_str = decode(new_img._label)
        return new_img

    def apply(self, operation: Callable[[npt.NDArray], npt.NDArray]) -> CovidImage:
        """
        Apply a Numpy operation on current instance and generate a new instance.

        :param operation: Operation to apply,
            should take one Numpy matrix as input and generate a Numpy matrix as output.
        :return: A new instance after the operation.
        """
        return CovidImage.from_np_array(operation(self._image), self._label)

    def save(self, image_path: Optional[str] = None) -> None:
        """
        Save the image to a file on disk.

        :param image_path: Path to destination image.
            If not set, would be the current ``image_path`` property.
            If set, will update ``image_path`` property of the current instance.
        :raises ValueError: In case the ``image_path`` parameter is in memory.
        """
        if image_path is None:
            image_path = self._image_path
        if image_path == IN_MEMORY_INDICATOR:
            raise ValueError("Cannot save to in-memory data.")
        self._image_path = image_path
        if image_path.endswith(".npy.xz"):
            io_helper.write_np_xz(self._image, image_path)
        else:
            skiio.imsave(image_path, self._image)


class CovidDataSet:
    """
    The COVID dataset abstraction is a dataset with following features:

    - Allows partial balanced/unbalanced image loading from local filesystem or from another dataset;
    - Supports parallel operation on images;
    - Interfaces to other machine-learning/deep-learning libraries like ``sklearn``;
    """
    _loaded_image: List[CovidImage]
    _loaded_image_with_label: Dict[int, List[CovidImage]]
    _dataset_path: str
    _sklearn_dataset: Optional[Tuple[npt.NDArray, npt.NDArray]]

    @property
    def dataset_path(self) -> str:
        """Read-only absolute path of datasets"""
        return self._dataset_path

    @property
    def sklearn_dataset(self) -> Tuple[npt.NDArray, npt.NDArray]:
        """
        Prepare and return cached dataset for ``sklearn``.

        :return: A tuple of ``X`` and ``y`` for :py:func:`fit`-like functions.
            For example, as is used in :external+sklearn:py:class:`sklearn.neighbors.KNeighborsClassifier`.
        """
        if self._sklearn_dataset is None:
            num_images = len(self._loaded_image)
            if num_images == 0:
                raise ValueError("Empty dataset!")
            _img_size = self._loaded_image[0].np_array.shape
            for img in self._loaded_image:
                if img.np_array.shape != _img_size:
                    raise ValueError(f"Image {img} have different size!")
            X:npt.NDArray = np.ndarray((num_images, operator.mul(*_img_size)), dtype=float)
            y:npt.NDArray = np.ndarray((num_images,), dtype=int)
            for i, img in enumerate(tqdm.tqdm(
                    iterable=self._loaded_image,
                    desc="Parsing to SKLearn..."
            )):
                X[i] = np.ravel(img.np_array)
                y[i] = img.label
            unique_labels, counts = np.unique(y, return_counts=True)
            unique_labels = [decode(label) for label in unique_labels]
            _lh.info("Loaded labels %s with corresponding counts %s", str(unique_labels), str(counts))
            self._sklearn_dataset = X, y
        return self._sklearn_dataset

    def __init__(self) -> None:
        """
        Dumb initializer which initializes a dataset for in-memory use.
        """
        self._loaded_image = []
        self._loaded_image_with_label = defaultdict(lambda: [])
        self._dataset_path = IN_MEMORY_INDICATOR
        self._sklearn_dataset = None

    def _load_impl(self, image_path: str) -> None:
        img = CovidImage.from_file(image_path)
        self._loaded_image.append(img)
        self._loaded_image_with_label[img.label].append(img)

    def _preload_hook(
            self,
            dataset_path: str,
            size: int = -1,
            balanced: bool = True
    ) -> List[str]:
        self._dataset_path = dataset_path
        _lh.info("Requested data from %s...", self._dataset_path)
        all_image_paths = list(reduce(
            operator.add,
            map(
                lambda ext: list(glob.glob(os.path.join(self._dataset_path, "*", f"*.{ext}"))),
                VALID_IMAGE_EXTENSIONS
            )
        ))
        image_paths_with_label = defaultdict(lambda: [])
        for image_path in tqdm.tqdm(
                iterable=all_image_paths,
                desc="Parsing image directory..."
        ):
            label = encode(resolve_label_from_path(image_path))
            image_paths_with_label[label].append(image_path)
        size = _get_max_size_helper(size, balanced, len(all_image_paths), image_paths_with_label)
        _lh.info("Loading %d data from %s...", size, self._dataset_path)

        if balanced:
            all_image_paths = list(itertools.chain(
                *list(map(
                    lambda image_paths: random.sample(image_paths, size // 3),
                    image_paths_with_label.values()
                ))
            ))
        else:
            all_image_paths = random.sample(all_image_paths, size)
        return all_image_paths

    @classmethod
    def from_directory(
            cls,
            dataset_path: str,
            size: int = -1,
            balanced: bool = True
    ) -> CovidDataSet:
        """
        Generate a new instance from directory of images.
        The directory (absolute path to ``sample_covid_image``) should be like following structure:

        .. code-block::

            sample_covid_image/
            ├── COVID-19
            │  ├── 0371ce19-4fca-4050-83bf-0b71113d2b60.png
            │  └── ea8bd845-4bb0-4444-ae92-21ed30ac4f1d.png
            ├── NORMAL
            │  ├── a5b7ffb8-2d5c-434d-bf94-87ef59a3cd86.png
            │  └── a9d0f8a0-0f68-4b56-910e-60be7a49a038.png
            └── Viral_Pneumonia
                ├── 0294df45-7b23-4870-9f16-374b62d9045c.png
                └── 07d3a378-54c9-4c5a-8f6b-09d1087473cf.png


            3 directories, 90 files


        :param dataset_path: Absolute path of the dataset-containing directory.
        :param size: Number of images needed to be loaded.
        :param balanced: Whether the loaded images should be "balanced" -- i.e., have same number of each category.
        """
        new_ds = cls()
        all_image_paths = new_ds._preload_hook(
            dataset_path=dataset_path,
            size=size,
            balanced=balanced
        )
        for image_path in tqdm.tqdm(
                all_image_paths,
                desc="Loading data..."
        ):
            new_ds._load_impl(image_path)
        _lh.info("Shuffling loaded data...")
        random.shuffle(new_ds._loaded_image)
        _lh.info("Finished loading data...")
        return new_ds

    @classmethod
    def parallel_from_directory(
            cls,
            dataset_path: str,
            size: int = -1,
            balanced: bool = True,
            **kwargs
    ) -> CovidDataSet:
        """
        Parallel version of :py:func:`from_directory`.
        """
        if kwargs.get("backend") is not None:
            raise ValueError("backend should not be set!")
        new_ds = cls()
        all_image_paths = new_ds._preload_hook(
            dataset_path=dataset_path,
            size=size,
            balanced=balanced
        )
        _ = list(joblib_helper.parallel_map(
            new_ds._load_impl,
            tqdm.tqdm(
                all_image_paths,
                desc="Loading data..."
            ),
            backend="threading",
            **kwargs
        ))
        _lh.info("Shuffling loaded data...")
        random.shuffle(new_ds._loaded_image)
        _lh.info("Finished loading data with %d images loaded", len(new_ds._loaded_image))
        return new_ds

    @classmethod
    def from_loaded_image(cls, loaded_image: List[CovidImage]) -> CovidDataSet:
        """
        Generate a new instance from a list of py:class:`CovidImage`.
        """
        new_ds = cls()
        new_ds._loaded_image = loaded_image
        for img in new_ds._loaded_image:
            new_ds._loaded_image_with_label[img.label].append(img)
        return new_ds

    def apply(
            self,
            operation: Callable[[npt.NDArray], npt.NDArray]
    ) -> CovidDataSet:
        """
        Apply a function to each image inside the dataset. See :py:func:`CovidImage.apply` for more details.

        :param operation: Operation to be applied to each image.
        :return: New dataset with image operated.
        """
        new_ds = CovidDataSet.from_loaded_image(list(map(
            lambda image: image.apply(operation),
            tqdm.tqdm(iterable=self._loaded_image, desc="Applying operations...")
        )))
        return new_ds

    def parallel_apply(
            self,
            operation: Callable[[npt.NDArray], npt.NDArray],
            **kwargs
    ) -> CovidDataSet:
        """
        Parallel version of :py:func:`apply`.
        """
        new_ds = CovidDataSet.from_loaded_image(list(joblib_helper.parallel_map(
            lambda image: image.apply(operation),
            tqdm.tqdm(iterable=self._loaded_image, desc="Applying operations..."),
            **kwargs
        )))
        return new_ds

    def sample(self, size: int = -1, balanced: bool = True) -> CovidDataSet:
        """
        Sample the current dataset and return the sampled dataset.

        :param size: Number of images needed to be loaded.
        :param balanced: Whether the loaded images should be "balanced" -- i.e., have same number of each category.
        """
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

    def _save_impl(
            self,
            img: CovidImage,
            dataset_path: str,
            extension: str
    ) -> None:
        if img.image_path == IN_MEMORY_INDICATOR:
            _image_path = ".".join((str(uuid.uuid4()), extension))
        else:
            _image_path = img.image_path
        img.save(os.path.join(dataset_path, img.label_str, os.path.basename(_image_path)))

    def _presave_hook(self, dataset_path: Optional[str]) -> None:
        if dataset_path is None:
            dataset_path = self._dataset_path
        if dataset_path == IN_MEMORY_INDICATOR:
            raise ValueError("Cannot save to in-memory data.")
        if os.path.exists(dataset_path):
            shutil.rmtree(dataset_path)
        os.makedirs(dataset_path, exist_ok=False)
        self._dataset_path = dataset_path
        for label in self._loaded_image_with_label.keys():
            label_str = decode(label)
            os.makedirs(os.path.join(self._dataset_path, label_str), exist_ok=False)

    def save(self, dataset_path: Optional[str] = None, extension: str = ".npy.xz") -> None:
        """
        Save the dataset to a file on disk.

        :param dataset_path: Path to destination image.
            If not set, would be the ``dataset_path`` property.
            If set, will update ``dataset_path`` property of the current instance.
        :param extension: The extension name of the image. should be in ``VALID_IMAGE_EXTENSIONS``.
        :raises ValueError: In case the ``dataset_path`` parameter is in memory.
        """
        self._presave_hook(dataset_path)
        _ = list(map(
            lambda img: self._save_impl(img, dataset_path, extension),
            tqdm.tqdm(
                iterable=list(itertools.chain(*self._loaded_image_with_label.values())),
                desc="Saving images..."
            )
        ))

    def parallel_save(self, dataset_path: Optional[str] = None, extension: str = ".npy.xz", **kwargs) -> None:
        """
        Parallel version of :py:func:`save`.
        """
        self._presave_hook(dataset_path)
        _ = list(joblib_helper.parallel_map(
            lambda img: self._save_impl(img, dataset_path, extension),
            tqdm.tqdm(
                iterable=list(itertools.chain(*self._loaded_image_with_label.values())),
                desc="Saving images..."
            ),
            **kwargs
        ))

    def __len__(self):
        return len(self._loaded_image)

    def __iter__(self) -> Iterable[CovidImage]:
        return iter(self._loaded_image)

    def __getitem__(self, i: int) -> CovidImage:
        return self._loaded_image[i]

    def __setitem__(self, i: int, value: CovidImage):
        self._loaded_image[i] = value


if __name__ == "__main__":
    doctest.testmod()
