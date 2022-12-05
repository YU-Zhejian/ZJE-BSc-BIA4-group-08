"""
Helper classes and functions to general-purposed machine-learning/deep-learning over COVID image dataset.
"""

from __future__ import annotations

__all__ = (
    "IN_MEMORY_INDICATOR",
    "VALID_IMAGE_EXTENSIONS",
    "CovidImage",
    "CovidDataSet",
    "resolve_label_str_from_path",
    "generate_fake_classification_dataset"
)

import glob
import itertools
import operator
import os
import random
import shutil
import uuid
from collections import defaultdict
from functools import reduce
from typing import Tuple, List, Dict, Callable, Iterable, Optional, Any, Mapping, Union, overload

import numpy as np
import numpy.typing as npt
import skimage
import skimage.draw as skidraw
import skimage.io as skiio
import skimage.transform as skitrans
import skimage.util as skiutil
import torch
import tqdm

from BIA_G8 import get_lh
from BIA_G8.helper import io_helper, joblib_helper, torch_helper, ndarray_helper, ml_helper
from BIA_G8.helper.ml_helper import MachinelearningDatasetInterface

_lh = get_lh(__name__)

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


def resolve_label_str_from_path(abspath: str) -> str:
    """
    Get image paths from label name.

    >>> resolve_label_str_from_path("/100/200/300.png")
    '200'

    :param abspath: Absolute path to the image.
    :return: Resolved label in string format.
    """
    return os.path.split(os.path.split(abspath)[0])[1]


def infer_encode_decode_from_filesystem(dataset_path: str) -> Tuple[
    int, Tuple[Callable[[str], int], Callable[[int], str]]]:
    _lh.info("Inferring encoder decoder from directory...")
    encoder_dict = {
        os.path.basename(image_dirname.rstrip(os.path.sep)): index
        for index, image_dirname in enumerate(glob.glob(os.path.join(dataset_path, "*", "")))
    }
    _lh.info("Encoder Dict: %s", repr(encoder_dict))
    return len(encoder_dict), ml_helper.generate_encoder_decoder(encoder_dict)


def _get_max_size_helper(
        size: int,
        n_classes: int,
        balanced: bool,
        full_size: int,
        image_with_label: Dict[Any, Any]
) -> int:
    if balanced:
        max_size = min(map(len, image_with_label.values())) * n_classes
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
    _np_array: npt.NDArray
    _torch_tensor: Optional[torch.Tensor]

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
        """Copied numpy aray of the image."""
        return self._np_array.copy()

    @property
    def torch_tensor(self) -> torch.Tensor:
        if self._torch_tensor is None:
            self._torch_tensor = torch.tensor(
                data=np.expand_dims(
                    ndarray_helper.scale_np_array(
                        self._np_array
                    ),
                    axis=0
                ),
                dtype=torch.float
            )
        return self._torch_tensor

    @property
    def image_path(self) -> str:
        """Read-only absolute path to the image."""
        return self._image_path

    def __init__(
            self,
            image: npt.NDArray,
            image_path: str,
            label: int,
            label_str: str
    ):
        self._np_array = image.copy()
        self._np_array.setflags(write=False)
        self._torch_tensor = None
        self._image_path = image_path
        self._label_str = label_str
        self._label = label

    @classmethod
    def from_file(
            cls,
            image_path: str,
            label: int,
            label_str: str
    ) -> CovidImage:
        """
        Generate a new instance from existing file.

        :param label: Label in integer
        :param label_str: Label in string
        :param image_path: Absolute path to the image.
        """
        if image_path.endswith("npy.xz"):
            image = io_helper.read_np_xz(image_path)
        else:
            image = skiio.imread(image_path)
        return cls(
            image=image,
            image_path=image_path,
            label=label,
            label_str=label_str
        )

    @classmethod
    def from_np_array(cls, image: npt.NDArray, label: int, label_str: str) -> CovidImage:
        """
        Generate a new instance based on existing Numpy array.

        :param label_str: Label in string.
        :param image: Image in numpy aray.
        :param label: Label in integer.
        """
        return cls(
            image=image,
            image_path=IN_MEMORY_INDICATOR,
            label=label,
            label_str=label_str
        )

    def apply(self, operation: Callable[[npt.NDArray], npt.NDArray]) -> CovidImage:
        """
        Apply a Numpy operation on current instance and generate a new instance.

        :param operation: Operation to apply,
            should take one Numpy matrix as input and generate a Numpy matrix as output.
        :return: A new instance after the operation.
        """
        return CovidImage.from_np_array(
            image=operation(self.np_array),
            label=self._label,
            label_str=self._label_str
        )

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
            io_helper.write_np_xz(self._np_array, image_path)
        else:
            skiio.imsave(image_path, self._np_array)


class CovidDataSet(MachinelearningDatasetInterface):
    """
    The COVID dataset abstraction is a dataset with following features:

    - Allows partial balanced/unbalanced image loading from local filesystem or from another dataset;
    - Supports parallel operation on images;
    - Interfaces to other machine-learning/deep-learning libraries like ``sklearn``;
    """
    _loaded_images: List[CovidImage]
    _loaded_images_with_labels: Dict[int, List[CovidImage]]
    _dataset_path: str
    _encode: Callable[[str], int]
    _decode: Callable[[int], str]
    _n_classes: int
    _torch_dataset: Optional[torch_helper.DictBackedTorchDataSet]
    _sklearn_dataset: Optional[Tuple[npt.NDArray, npt.NDArray]]

    @property
    def dataset_path(self) -> str:
        """Read-only absolute path of datasets"""
        return self._dataset_path

    @property
    def encode(self) -> Callable[[str], int]:
        """Read-only encoder function"""
        return self._encode

    @property
    def decode(self) -> Callable[[int], str]:
        """Read-only decoder function"""
        return self._decode

    @property
    def n_classes(self) -> int:
        return self._n_classes

    @property
    def sklearn_dataset(self) -> Tuple[npt.NDArray, npt.NDArray]:
        if self._sklearn_dataset is None:
            num_images = len(self._loaded_images)
            if num_images == 0:
                raise ValueError("Empty dataset!")
            _img_size = self._loaded_images[0].np_array.shape
            for img in self._loaded_images:
                if img.np_array.shape != _img_size:
                    raise ValueError(f"Image {img} have different size!")
            x: npt.NDArray = np.ndarray((num_images, operator.mul(*_img_size)), dtype=float)
            y: npt.NDArray = np.ndarray((num_images,), dtype=int)
            for i, img in enumerate(tqdm.tqdm(
                    iterable=self._loaded_images,
                    desc="Parsing to SKLearn..."
            )):
                x[i] = np.ravel(img.np_array)
                y[i] = img.label
            unique_labels, counts = np.unique(y, return_counts=True)
            unique_labels_str = [self.decode(label) for label in unique_labels]
            _lh.info("Loaded labels %s with corresponding counts %s", unique_labels_str, str(counts))
            self._sklearn_dataset = x, y
        return self._sklearn_dataset

    @property
    def torch_dataset(self) -> torch_helper.DictBackedTorchDataSet:
        if self._torch_dataset is None:
            self._torch_dataset = torch_helper.DictBackedTorchDataSet({
                index: image
                for index, image in enumerate(
                    (covid_img.torch_tensor, torch.tensor(covid_img.label).long())
                    for covid_img in tqdm.tqdm(
                        iterable=self._loaded_images,
                        desc="Parsing to Torch..."
                    )
                )
            })
        return self._torch_dataset

    def train_test_split(self, ratio: float = 0.7) -> Tuple[CovidDataSet, CovidDataSet]:
        result = np.random.binomial(1, ratio, len(self))

        train_images = [self._loaded_images[index] for index in np.where(result == 1)[0]]
        test_images = [self._loaded_images[index] for index in np.where(result == 0)[0]]
        return (
            CovidDataSet.from_loaded_images(
                loaded_images=train_images,
                encode=self.encode,
                decode=self.decode,
                n_classes=self.n_classes
            ),
            CovidDataSet.from_loaded_images(
                loaded_images=test_images,
                encode=self.encode,
                decode=self.decode,
                n_classes=self.n_classes
            )
        )

    def __init__(
            self,
            encode: Callable[[str], int],
            decode: Callable[[int], str],
            n_classes: int
    ) -> None:
        """
        Dumb initializer which initializes a dataset for in-memory use.
        """
        self._n_classes = n_classes
        self._encode = encode
        self._decode = decode
        self._loaded_images = []
        self._loaded_images_with_labels = defaultdict(lambda: [])
        self._dataset_path = IN_MEMORY_INDICATOR
        self._sklearn_dataset = None
        self._torch_dataset = None

    def _load_impl(self, image_path: str) -> None:
        label_str = resolve_label_str_from_path(image_path)
        img = CovidImage.from_file(
            image_path,
            label=self.encode(label_str),
            label_str=label_str,
        )
        self._loaded_images.append(img)
        self._loaded_images_with_labels[img.label].append(img)

    def _preload_hook(
            self,
            dataset_path: str,
            size: int,
            balanced: bool
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
            label = self.encode(resolve_label_str_from_path(image_path))
            image_paths_with_label[label].append(image_path)
        size = _get_max_size_helper(
            size,
            self.n_classes,
            balanced,
            len(all_image_paths),
            image_paths_with_label
        )
        _lh.info("Loading %d data from %s...", size, self._dataset_path)

        if balanced:
            all_image_paths = list(itertools.chain(
                *list(map(
                    lambda image_paths: random.sample(image_paths, size // self.n_classes),
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
            balanced: bool = True,
            encode: Optional[Callable[[str], int]] = None,
            decode: Optional[Callable[[int], str]] = None,
            n_classes: Optional[int] = None
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

        if encode is None:
            n_classes, (encode, decode) = infer_encode_decode_from_filesystem(dataset_path)
        new_ds = cls(encode=encode, decode=decode, n_classes=n_classes)
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
        random.shuffle(new_ds._loaded_images)
        _lh.info("Finished loading data...")
        return new_ds

    @classmethod
    def parallel_from_directory(
            cls,
            dataset_path: str,
            size: int = -1,
            balanced: bool = True,
            encode: Optional[Callable[[str], int]] = None,
            decode: Optional[Callable[[int], str]] = None,
            n_classes: Optional[int] = None,
            joblib_kwds: Optional[Mapping[str, Any]] = None
    ) -> CovidDataSet:
        """
        Parallel version of :py:func:`from_directory`.
        """
        if joblib_kwds is None:
            joblib_kwds = {}
        joblib_kwds = dict(joblib_kwds)
        if encode is None:
            n_classes, (encode, decode) = infer_encode_decode_from_filesystem(dataset_path)
        new_ds = cls(encode=encode, decode=decode, n_classes=n_classes)
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
            **joblib_kwds
        ))
        _lh.info("Shuffling loaded data...")
        random.shuffle(new_ds._loaded_images)
        _lh.info("Finished loading data with %d images loaded", len(new_ds._loaded_images))
        return new_ds

    @classmethod
    def from_loaded_images(
            cls,
            loaded_images: Iterable[CovidImage],
            encode: Callable[[str], int],
            decode: Callable[[int], str],
            n_classes: int
    ) -> CovidDataSet:
        """
        Generate a new instance from a list of py:class:`CovidImage`.
        """
        new_ds = cls(encode=encode, decode=decode, n_classes=n_classes)
        new_ds._loaded_images = list(loaded_images)
        for img in new_ds._loaded_images:
            new_ds._loaded_images_with_labels[img.label].append(img)
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
        new_ds = CovidDataSet.from_loaded_images(
            list(map(
                lambda image: image.apply(operation),
                tqdm.tqdm(iterable=self._loaded_images, desc="Applying operations...")
            )),
            encode=self.encode,
            decode=self.decode,
            n_classes=self.n_classes
        )
        return new_ds

    def parallel_apply(
            self,
            operation: Callable[[npt.NDArray], npt.NDArray],
            joblib_kwds: Optional[Mapping[str, Any]] = None,
            desc: str = "Applying operations..."
    ) -> CovidDataSet:
        """
        Parallel version of :py:func:`apply`.
        """
        if joblib_kwds is None:
            joblib_kwds = {}
        new_ds = CovidDataSet.from_loaded_images(
            joblib_helper.parallel_map(
                lambda image: image.apply(operation),
                tqdm.tqdm(iterable=self._loaded_images, desc=desc),
                **joblib_kwds
            ),
            encode=self.encode,
            decode=self.decode,
            n_classes=self.n_classes
        )
        return new_ds

    def sample(self, size: int = -1, balanced: bool = True) -> CovidDataSet:
        """
        Sample the current dataset and return the sampled dataset.

        :param size: Number of images needed to be loaded.
        :param balanced: Whether the loaded images should be "balanced" -- i.e., have same number of each category.
        """

        _lh.info("Sampling data...")
        size = _get_max_size_helper(
            size,
            self.n_classes,
            balanced,
            len(self._loaded_images),
            self._loaded_images_with_labels
        )
        _lh.info("Loading %d data from %s...", size, self._dataset_path)
        sampled_images = []
        if balanced:
            for image_catagory in self._loaded_images_with_labels.values():
                for img in random.sample(image_catagory, size // self.n_classes):
                    sampled_images.append(img)
        else:
            for img in random.sample(self._loaded_images, size):
                sampled_images.append(img)
        _lh.info("Shuffling loaded data...")
        random.shuffle(sampled_images)
        _lh.info("Finished loading data...")
        new_ds = CovidDataSet.from_loaded_images(
            loaded_images=sampled_images,
            encode=self.encode,
            decode=self.decode,
            n_classes=self.n_classes
        )
        return new_ds

    @staticmethod
    def _save_impl(
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
        for label in self._loaded_images_with_labels.keys():
            label_str = self.decode(label)
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
                iterable=list(itertools.chain(*self._loaded_images_with_labels.values())),
                desc="Saving images..."
            )
        ))

    def parallel_save(
            self,
            dataset_path: Optional[str] = None,
            extension: str = ".npy.xz",
            joblib_kwds: Optional[Mapping[str, Any]] = None
    ) -> None:
        """
        Parallel version of :py:func:`save`.
        """
        if joblib_kwds is None:
            joblib_kwds = {}
        self._presave_hook(dataset_path)
        _ = list(joblib_helper.parallel_map(
            lambda img: self._save_impl(img, dataset_path, extension),
            tqdm.tqdm(
                iterable=list(itertools.chain(*self._loaded_images_with_labels.values())),
                desc="Saving images..."
            ),
            **joblib_kwds
        ))

    def __len__(self):
        return len(self._loaded_images)

    def __iter__(self) -> Iterable[CovidImage]:
        return iter(self._loaded_images)

    @overload
    def __getitem__(self, i: int) -> CovidImage:
        ...

    @overload
    def __getitem__(self, s: slice) -> CovidDataSet:
        ...

    def __getitem__(self, i: Union[int, slice]) -> Union[CovidImage, CovidDataSet]:
        if isinstance(i, slice):
            retl = []
            start, stop, step = i.start, i.stop, i.step
            if step is None:
                step = 1
            if stop is None or stop == -1:
                stop = len(self)
            if start is None:
                start = 0
            for _i in range(start, stop, step):
                retl.append(self[_i])
            return self.from_loaded_images(
                loaded_images=retl,
                encode=self.encode,
                decode=self.decode,
                n_classes=self.n_classes
            )
        return self._loaded_images[i]

    def __setitem__(self, i: int, value: CovidImage):
        self._loaded_images[i] = value


def generate_fake_classification_dataset(
        size: int = 120,
        n_classes: int = 4,
        width: int = 256,
        height: int = 256
) -> CovidDataSet:
    labels = ["stride", "circle", "square", "blank"][0:n_classes]
    _encoder_dict = {k: v for k, v in zip(labels, range(len(labels)))}
    encode, decode = ml_helper.generate_encoder_decoder(_encoder_dict)

    blank = np.zeros((100, 100), dtype=int)
    stride = skimage.img_as_int(np.array(reduce(operator.add, map(lambda x: [[x] * 100] * 10, range(0, 100, 10)))))

    circle = blank.copy()
    rr, cc = skidraw.circle_perimeter(r=50, c=50, radius=30)
    circle[rr, cc] = 100
    circle = skimage.img_as_int(circle)

    square = blank.copy()
    rr, cc = skidraw.rectangle(start=(20, 20), extent=(60, 60))
    square[rr, cc] = 100
    square = skimage.img_as_int(square)
    images = [
        CovidImage.from_np_array(img, label, decode(label))
        for label, img in enumerate((stride, circle, square, blank))
    ]
    return CovidDataSet.from_loaded_images(
        itertools.chain(*itertools.repeat(images, size // n_classes)),
        encode=encode,
        decode=decode,
        n_classes=len(labels)
    ).parallel_apply(
        lambda img: skimage.img_as_int(
            skiutil.random_noise(
                skimage.img_as_float(img),
                mode="pepper"
            )
        ),
        desc="Adding pepper noise..."
    ).parallel_apply(
        lambda img: skimage.img_as_int(
            skitrans.rotate(
                img,
                random.random() * 120 - 60
            )
        ),
        desc="Rotating random degree..."
    ).parallel_apply(
        lambda img: skitrans.resize(
            img,
            (width, height)
        ),
        desc="Scaling to wanted size..."
    )
