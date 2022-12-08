"""
The upscalers would upscale the image by deep learning.
"""

from __future__ import annotations

import gc
import os
from abc import abstractmethod
from statistics import mean
from typing import Iterable, Dict, Any, Final

import numpy as np
import numpy.typing as npt
import skimage
import skimage.color as skicol
import skimage.transform as skitrans
import torch
import torch.utils.data as tud
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error
from torch import nn

from BIA_G8 import get_lh
from BIA_G8.data_analysis.covid_dataset_configuration import CovidDatasetConfiguration
from BIA_G8.helper.io_helper import SerializableInterface, write_tensor_xz, write_toml_with_metadata, \
    read_toml_with_metadata, read_tensor_xz
from BIA_G8.helper.ml_helper import MachinelearningDatasetInterface
from BIA_G8.helper.ndarray_helper import scale_np_array
from BIA_G8.torch_modules.scgan import SCGANGenerator, SCGANDiscriminator, TruncatedVGG19

_lh = get_lh(__name__)


class UpscalerInterface(SerializableInterface):
    """
    Abstract Model for extension

    The model supports pipelining like ``absu().fit(train_data).evaluate(test_data)``
    """
    name: str
    """
    Class attribute that represents human readable classifier name.

    :meta private:
    """

    description: str
    """
    Class attribute that represents human readable classifier description.

    :meta private:
    """

    @abstractmethod
    def fit(self, dataset: MachinelearningDatasetInterface) -> UpscalerInterface:
        """
        Train the module using :py:class:`MachinelearningDatasetInterface`.

        :param dataset: The training set.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, image: npt.NDArray) -> npt.NDArray:
        """
        Predict over one image.

        :param image: Input image, which should be an image of npt.NDArray[float64] datatype.
            at range ``[0, 1]``.
        :return: Predicted upscaled image.
        """
        raise NotImplementedError

    @abstractmethod
    def predicts(self, images: Iterable[npt.NDArray]) -> Iterable[npt.NDArray]:
        """
        Predict over a batch of images. See :py:func:`UpscalerInterface.predict()`
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def new(cls, **params):
        """
        Initialize the

        :param params: Possible parameters.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls, path: str, load_model: bool = True):
        """
        Load the model from TOML

        :param path: Source TOML path.
        :param load_model: Whether to load pretrained model (if exist).
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str, save_model: bool = True) -> None:
        """
        Save the model to TOML.

        :param path: Destination TOML path.
        :param save_model: Whether to save the trained model.
        """
        raise NotImplementedError


class SCGANUpscaler(UpscalerInterface):
    _generator: SCGANGenerator
    _discriminator: SCGANDiscriminator
    _truncated_vgg19: TruncatedVGG19
    _lr: float
    _device: str
    _batch_size: int
    _num_epochs: int
    _beta: float
    _scale_factor: int
    _generator_params: Dict[str, Any]
    _discriminator_params: Dict[str, Any]
    _truncated_vgg19_params: Dict[str, Any]
    _hyper_params: Dict[str, Any]
    name: Final[str] = "SCGAN"
    description: Final[str] = "N/A"

    @staticmethod
    def _convert_to_3_channels(batched_input: torch.Tensor) -> torch.Tensor:
        """
        ``[BATCH_SIZE, 1, WID, HEIGHT] -> [BATCH_SIZE, 3, WID, HEIGHT]``
        """
        images_in_3_channels = []
        for img in batched_input:
            images_in_3_channels.extend(torch.stack([img, img, img], dim=1))
        return torch.stack(images_in_3_channels, dim=0)

    @staticmethod
    def _decrease_resolution_3_channel(
            batched_input: torch.Tensor,
            scale_factor: int
    ) -> torch.Tensor:
        """
        ``[BATCH_SIZE, 1, WID, HEIGHT] -> [BATCH_SIZE, 3, WID // SF, HEIGHT // SF]``
        """
        downscaled_images = []
        for img in batched_input:
            img = np.moveaxis(img.numpy(), 0, -1)
            downscaled_image = np.expand_dims(np.moveaxis(
                skitrans.resize(
                    img,
                    (img.shape[0] // scale_factor, img.shape[1] // scale_factor)
                ),
                -1, 0
            ), axis=0)
            downscaled_images.extend(torch.tensor(downscaled_image))
        return torch.stack(downscaled_images, dim=0)

    def fit(self, dataset: MachinelearningDatasetInterface) -> SCGANUpscaler:
        optimizer_g = torch.optim.Adam(
            params=filter(lambda p: p.requires_grad, self._generator.parameters()),
            lr=self._lr
        )
        optimizer_d = torch.optim.Adam(
            params=filter(lambda p: p.requires_grad, self._discriminator.parameters()),
            lr=self._lr
        )
        content_loss_criterion = nn.MSELoss().to(self._device)
        adversarial_loss_criterion = nn.BCEWithLogitsLoss().to(self._device)

        train_data_loader = tud.DataLoader(dataset.torch_dataset, batch_size=self._batch_size, shuffle=True)
        for epoch in range(self._num_epochs):
            for i, (high_res_images, _) in enumerate(train_data_loader):

                high_res_images = self._convert_to_3_channels(high_res_images)
                low_res_images = self._decrease_resolution_3_channel(high_res_images, self._scale_factor)
                low_res_images, high_res_images = low_res_images.to(self._device), high_res_images.to(self._device)
                generated_images = self._generator(low_res_images)

                generated_images_vgg = self._truncated_vgg19(generated_images)
                high_res_images_vgg = self._truncated_vgg19(high_res_images).detach()
                content_loss = content_loss_criterion(generated_images_vgg, high_res_images_vgg)

                generated_images_discriminated = self._discriminator(generated_images)
                adversarial_loss = adversarial_loss_criterion(
                    generated_images_discriminated,
                    torch.ones_like(generated_images_discriminated)
                )
                perceptual_loss = content_loss + self._beta * adversarial_loss
                optimizer_g.zero_grad()
                perceptual_loss.backward()
                optimizer_g.step()

                high_res_images_discriminated = self._discriminator(high_res_images)
                generated_images_discriminated = self._discriminator(generated_images).detach()
                adversarial_loss = adversarial_loss_criterion(
                    generated_images_discriminated,
                    torch.zeros_like(generated_images_discriminated)
                ) + adversarial_loss_criterion(
                    high_res_images_discriminated,
                    torch.ones_like(high_res_images_discriminated)
                )
                optimizer_d.zero_grad()
                adversarial_loss.backward()
                optimizer_d.step()

                del (
                    generated_images_discriminated,
                    high_res_images_discriminated,
                    high_res_images_vgg,
                    generated_images_vgg,
                    low_res_images
                )
                gc.collect()

                if i % 10 == 0:
                    high_res_images_np = high_res_images.detach().cpu().numpy()
                    generated_images_np = generated_images.detach().cpu().numpy()
                    psnrs = []
                    ssims = []
                    accus = []
                    mses = []
                    for high_res_image, generated_image in zip(high_res_images_np, generated_images_np):
                        high_res_image = np.moveaxis(high_res_image, 0, -1)
                        high_res_image = skimage.img_as_ubyte(high_res_image)
                        high_res_image = skicol.rgb2gray(high_res_image)
                        generated_image = np.moveaxis(generated_image, 0, -1)
                        generated_image = scale_np_array(generated_image, domain=(-1, 1))
                        generated_image = skimage.img_as_ubyte(generated_image)
                        generated_image = skicol.rgb2gray(generated_image)
                        # print(describe(high_res_image), describe(generated_image))
                        psnrs.append(peak_signal_noise_ratio(high_res_image, generated_image))
                        ssims.append(structural_similarity(high_res_image, generated_image))
                        mses.append(mean_squared_error(high_res_image, generated_image))
                        accus.append(
                            (
                                    np.sum(high_res_image == generated_image) //
                                    (high_res_image.shape[0] * high_res_image.shape[1])
                            ).item()
                        )

                    _lh.info(
                        "%s Epoch %d batch %d: absolute accuracy %.2f%%, PSNR %.2f, SSIM %.2f, MSE %.2f",
                        self.__class__.__name__, epoch, i, mean(accus) * 100, mean(psnrs), mean(ssims), mean(mses)
                    )
        return self

    def predict(self, image: npt.NDArray) -> npt.NDArray:
        return next(self.predicts([image]))

    def predicts(self, images: Iterable[npt.NDArray]) -> Iterable[npt.NDArray]:
        images_torch = list(map(
            lambda image: torch.tensor(
                data=np.expand_dims(
                    scale_np_array(image),
                    axis=0
                ),
                dtype=torch.float
            ),
            images
        ))
        images_batch = torch.stack(images_torch, dim=0)
        images_batch = self._convert_to_3_channels(images_batch)

        with torch.no_grad():
            return map(
                lambda torch_img: skicol.rgb2gray(np.moveaxis(torch_img, 0, -1)),
                self._generator(images_batch.to(self._device)).cpu().detach().numpy()
            )

    def __init__(
            self,
            *,
            generator: SCGANGenerator,
            discriminator: SCGANDiscriminator,
            truncated_vgg19: TruncatedVGG19,
            generator_params: Dict[str, Any],
            discriminator_params: Dict[str, Any],
            truncated_vgg19_params: Dict[str, Any],
            hyper_params: Dict[str, Any]
    ):
        self._generator_params = generator_params
        self._discriminator_params = discriminator_params
        self._truncated_vgg19_params = truncated_vgg19_params
        self._hyper_params = hyper_params

        self._lr = self._hyper_params["lr"]
        self._device = self._hyper_params["device"]
        self._batch_size = self._hyper_params["batch_size"]
        self._num_epochs = self._hyper_params["num_epochs"]
        self._beta = self._hyper_params["beta"]
        self._scale_factor = self._hyper_params["scale_factor"]

        self._generator = generator.to(self._device)
        self._discriminator = discriminator.to(self._device)
        self._truncated_vgg19 = truncated_vgg19.to(self._device)

    @classmethod
    def new(
            cls,
            *,
            generator_params: Dict[str, Any],
            discriminator_params: Dict[str, Any],
            truncated_vgg19_params: Dict[str, Any],
            hyper_params: Dict[str, Any]
    ):
        return cls(
            generator=SCGANGenerator(**generator_params),
            discriminator=SCGANDiscriminator(**discriminator_params),
            truncated_vgg19=TruncatedVGG19(**truncated_vgg19_params),
            generator_params=generator_params,
            discriminator_params=discriminator_params,
            truncated_vgg19_params=truncated_vgg19_params,
            hyper_params=hyper_params
        )

    @classmethod
    def load(cls, path: str, load_model: bool = True):
        loaded_data = read_toml_with_metadata(path)
        if "generator_path" in loaded_data and load_model:
            _lh.info("%s: Loading pretrained model...", cls.__name__)
            return cls(
                hyper_params=loaded_data["hyper_params"],
                generator_params=loaded_data["generator_params"],
                discriminator_params=loaded_data["discriminator_params"],
                truncated_vgg19_params=loaded_data["truncated_vgg19_params"],
                generator=read_tensor_xz(loaded_data["generator_path"]),
                discriminator=read_tensor_xz(loaded_data["discriminator_path"]),
                truncated_vgg19=read_tensor_xz(loaded_data["truncated_vgg19_path"]),
            )

        else:
            _lh.info("%s: Loading parameters only...", cls.__name__)
            return cls.new(
                hyper_params=loaded_data["hyper_params"],
                generator_params=loaded_data["generator_params"],
                discriminator_params=loaded_data["discriminator_params"],
                truncated_vgg19_params=loaded_data["truncated_vgg19_params"]
            )

    def save(self, path: str, save_model: bool = True) -> None:
        _lh.info("%s: Saving...", self.__class__.__name__)
        path = os.path.abspath(path)
        out_dict = {
            "name": self.name,
            "hyper_params": self._hyper_params,
            "generator_params": self._generator_params,
            "discriminator_params": self._discriminator_params,
            "truncated_vgg19_params": self._truncated_vgg19_params
        }
        if save_model:
            generator_path = path + ".generator.pt.xz"
            write_tensor_xz(self._generator, generator_path)
            out_dict["generator_path"] = generator_path

            discriminator_path = path + ".discriminator.pt.xz"
            write_tensor_xz(self._generator, discriminator_path)
            out_dict["discriminator_path"] = discriminator_path

            truncated_vgg19_path = path + ".truncated_vgg19.pt.xz"
            write_tensor_xz(self._generator, truncated_vgg19_path)
            out_dict["truncated_vgg19_path"] = truncated_vgg19_path
        write_toml_with_metadata(out_dict, path)


if __name__ == "__main__":
    SCGANUpscaler.new(
        generator_params={
            "large_kernel_size": 9,
            "small_kernel_size": 3,
            "n_intermediate_channels": 64,
            "n_blocks": 16,
            "scale_factor": 2,
            "in_channels": 3
        },
        discriminator_params={
            "kernel_size": 3,
            "n_channels": 64,
            "n_blocks": 8,
            "fc_size": 1024,
            "in_channels": 3
        },
        truncated_vgg19_params={
            "i": 5,
            "j": 4
        },
        hyper_params={
            "num_epochs": 130,
            "lr": 0.0001,
            "device": "cuda",
            "batch_size": 16,
            "beta": 0.001,
            "scale_factor": 2
        }
    ).fit(
        dataset=CovidDatasetConfiguration.load(
            "ds_old.toml"
        ).dataset.parallel_apply(
            lambda img: skitrans.resize(
                img,
                (128, 128)
            ),
            desc="Scaling to wanted size..."
        )
    ).save("scgan.toml")
