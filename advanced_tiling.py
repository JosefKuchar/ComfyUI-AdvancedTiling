"""
Main advanced tiling implementation
"""

from typing import Optional
import functools
import copy

import torch
from torch import Tensor
from torch.nn import Conv2d
from torch.nn import functional as F
from torch.nn.modules.utils import _pair
from .modes import modes, Settings


@functools.cache
def calculate_mapping(
    original_size: tuple[int, int], padded_size: tuple[int, int], settings: Settings
):
    """
    Calculate mapping for pixels outside of the mask

    :param original_size: Original size of the image
    :param padded_size: Padded size of the image
    :param settings: Tiling settings
    :return: Mapping of pixels
    """

    mapping = []
    for y in range(padded_size[1]):
        for x in range(padded_size[0]):
            (new_x, new_y) = settings.tiling_fn(
                x, y, original_size, padded_size, settings
            )
            mapping.append([x, y, new_x, new_y])
    return list(zip(*mapping))


@functools.cache
def create_crop_mask(width: int, height: int, settings: Settings):
    """
    Crop image based on tiling settings

    :param image: Image to crop
    :param settings: Tiling settings
    :return: Cropped image
    """

    mask = torch.zeros((1, height, width, 1), dtype=torch.float32)
    for y in range(height):
        for x in range(width):
            # Calculate new coordinates
            (new_x, new_y) = settings.tiling_fn(
                x, y, (width, height), (width, height), settings
            )

            # If coordinates match, it means we are in the mask
            if new_x == x and new_y == y:
                mask[:, y, x] = 1
    return mask


def patch_model(model, settings: Settings):
    """
    Patch model to perform tiling - in place!

    :param model: Model to patch
    :param settings: Tiling settings
    """

    # Patch all Conv2d layers
    for layer in [layer for layer in model.modules() if isinstance(layer, Conv2d)]:
        # pylint: disable=protected-access, no-value-for-parameter
        layer._conv_forward = tiling_conv.__get__(layer, Conv2d)
        layer.tiling_settings = settings
    return


def tiling_conv(self, input_tensor: Tensor, weight: Tensor, bias: Optional[Tensor]):
    """
    Patched Conv2D forward function for tiling

    :param input_tensor: Input tensor
    :param weight: Weight tensor
    :param bias: Bias tensor
    :return: Convolution result
    """

    # Pad input tensor
    padded = F.pad(
        input_tensor,
        # pylint: disable=protected-access
        self._reversed_padding_repeated_twice,
    )
    # Calculate mapping
    mapping = calculate_mapping(
        (input_tensor.shape[-1], input_tensor.shape[-2]),
        (padded.shape[-1], padded.shape[-2]),
        self.tiling_settings,
    )
    # Apply tiling
    padded[:, :, mapping[1], mapping[0]] = padded[:, :, mapping[3], mapping[2]]
    # Perform convolution
    # pylint: disable=not-callable
    return F.conv2d(
        padded, weight, bias, self.stride, _pair(0), self.dilation, self.groups
    )


class AdvancedTilingSettings:
    """
    Tiling settings node that outputs tiling settings for other nodes
    """

    # pylint: disable=invalid-name

    @classmethod
    def INPUT_TYPES(cls):
        """
        Input types for the node
        """

        return {
            "required": {
                "mode": (list(modes.keys()),),
                "rotation": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 360.0, "step": 0.01},
                ),
            },
        }

    RETURN_TYPES = ("ADVANCED_TILING_SETTINGS",)
    RETURN_NAMES = ("SETTINGS",)
    FUNCTION = "run"

    def run(self, mode, rotation):
        """
        Creates tiling settings from node inputs
        """

        settings = Settings(mode, rotation)

        return (settings,)


class AdvancedTiling:
    """
    Patches Conv2D layers in a model to perform tiling
    """

    # pylint: disable=invalid-name

    @classmethod
    def INPUT_TYPES(cls):
        """
        Input types for the node
        """

        return {
            "required": {
                "settings": ("ADVANCED_TILING_SETTINGS",),
                "model": ("MODEL",),
            },
        }

    CATEGORY = "conditioning"
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "run"

    def run(self, settings, model):
        """
        Does the actual patching of the model
        """

        model_copy = copy.deepcopy(model)
        patch_model(model_copy.model, settings)

        return (model_copy,)


class AdvancedTilingVAEDecode:
    """
    Input types for the node
    """

    # pylint: disable=invalid-name

    @classmethod
    def INPUT_TYPES(cls):
        """
        Input types for the node
        """

        return {
            "required": {
                "settings": ("ADVANCED_TILING_SETTINGS",),
                "samples": ("LATENT",),
                "vae": ("VAE",),
                "crop": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "latent"

    def run(self, settings, samples, vae, crop):
        """
        Decode latents to image with tiling
        Optionally crop the image based on tiling settings

        :param settings: Tiling settings
        :param samples: Latent samples
        :param vae: VAE model
        :param crop: Whether to crop the image
        :return: Final image
        """

        vae_copy = copy.deepcopy(vae)
        # Enable tiling
        patch_model(vae_copy.first_stage_model, settings)
        # Decode latents to image
        image = vae_copy.decode(samples["samples"])
        if crop:
            # Crop image based on tiling settings
            mask = create_crop_mask(image.shape[2], image.shape[1], settings)
            image = torch.cat((image, mask), dim=3)

        return (image,)
