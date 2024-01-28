from logging import getLogger
from typing import Any, Literal

import numpy as np
import torch
from diffusers.schedulers.scheduling_utils import SchedulerOutput
from torch import FloatTensor, Tensor

from ...server.context import ServerContext

logger = getLogger(__name__)


class SchedulerPatch:
    server: ServerContext
    text_pipeline: bool
    wrapped: Any

    def __init__(self, server: ServerContext, scheduler: Any, text_pipeline: bool):
        self.server = server
        self.wrapped = scheduler
        self.text_pipeline = text_pipeline

    def __getattr__(self, attr):
        return getattr(self.wrapped, attr)

    def step(
        self, model_output: FloatTensor, timestep: Tensor, sample: FloatTensor
    ) -> SchedulerOutput:
        result = self.wrapped.step(model_output, timestep, sample)

        if self.text_pipeline and self.server.has_feature("mirror-latents"):
            logger.info("using experimental latent mirroring")

            if self.server.has_feature("mirror-latents-vertical"):
                axis_of_symmetry = 2
                expand_dims = (0, 1, 3)
            else:
                axis_of_symmetry = 3
                expand_dims = (0, 1, 2)

            white_point = 2
            black_point = result.prev_sample.shape[axis_of_symmetry] // 4
            center_line = result.prev_sample.shape[axis_of_symmetry] // 2

            gradient = linear_gradient(
                white_point, black_point, center_line, expand_dims
            )
            latents = result.prev_sample.numpy()

            # gradiated_latents = np.multiply(latents, gradient)
            inverse_gradiated_latents = np.multiply(
                np.flip(latents, axis=axis_of_symmetry), gradient
            )
            latents += inverse_gradiated_latents

            mask = np.ones_like(latents).astype(np.float32)
            # gradiated_mask = np.multiply(mask, gradient)
            # flipping the mask would do nothing, we need to flip the gradient for this one
            inverse_gradiated_mask = np.multiply(
                mask, np.flip(gradient, axis=axis_of_symmetry)
            )
            mask += inverse_gradiated_mask

            latents = np.where(mask > 0, latents / mask, latents)

            return SchedulerOutput(
                prev_sample=torch.from_numpy(latents),
            )
        else:
            return result


def linear_gradient(
    white_point: int,
    black_point: int,
    center_line: int,
    expand_dims: tuple[int, ...] = (0, 1, 2),
) -> np.ndarray:
    gradient = np.linspace(1, 0, black_point - white_point).astype(np.float32)
    gradient = np.pad(gradient, (white_point, 0), mode="constant", constant_values=1)
    gradient = np.pad(gradient, (0, center_line - black_point), mode="constant")
    gradient = np.reshape([gradient, np.flip(gradient)], -1)
    return np.expand_dims(gradient, expand_dims)


def mirror_latents(
    latents: np.ndarray,
    gradient: np.ndarray,
    center_line: int,
    direction: Literal["horizontal", "vertical"],
) -> np.ndarray:
    if direction == "horizontal":
        pad_left = max(0, -center_line)
        pad_right = max(0, 2 * center_line - latents.shape[3])

        # create the symmetrical copies
        padded_array = np.pad(
            latents, ((0, 0), (0, 0), (0, 0), (pad_left, pad_right)), mode="constant"
        )
        flipped_array = np.flip(padded_array, axis=3)

        # apply the gradient to both copies
        padded_gradiated = np.multiply(padded_array, gradient)
        flipped_gradiated = np.multiply(flipped_array, gradient)

        # produce masks
        mask = np.ones_like(latents).astype(np.float32)
        padded_mask = np.pad(
            mask, ((0, 0), (0, 0), (0, 0), (pad_left, pad_right)), mode="constant"
        )
        flipped_mask = np.flip(padded_mask, axis=3)

        padded_mask += np.multiply(padded_mask, gradient)
        padded_mask += np.multiply(flipped_mask, gradient)

        # combine the two copies
        result = padded_array + padded_gradiated + flipped_gradiated
        result = np.where(padded_mask > 0, result / padded_mask, result)
        return result[:, :, :, pad_left : pad_left + latents.shape[3]]
    elif direction == "vertical":
        pad_top = max(0, -center_line)
        pad_bottom = max(0, 2 * center_line - latents.shape[2])

        # create the symmetrical copies
        padded_array = np.pad(
            latents, ((0, 0), (0, 0), (pad_top, pad_bottom), (0, 0)), mode="constant"
        )
        flipped_array = np.flip(padded_array, axis=2)

        # apply the gradient to both copies
        padded_gradiated = np.multiply(
            padded_array.transpose(0, 1, 3, 2), gradient
        ).transpose(0, 1, 3, 2)
        flipped_gradiated = np.multiply(
            flipped_array.transpose(0, 1, 3, 2), gradient
        ).transpose(0, 1, 3, 2)

        # produce masks
        mask = np.ones_like(latents).astype(np.float32)
        padded_mask = np.pad(
            mask, ((0, 0), (0, 0), (pad_top, pad_bottom), (0, 0)), mode="constant"
        )
        flipped_mask = np.flip(padded_mask, axis=2)

        padded_mask += np.multiply(
            padded_mask.transpose(0, 1, 3, 2), gradient
        ).transpose(0, 1, 3, 2)
        padded_mask += np.multiply(
            flipped_mask.transpose(0, 1, 3, 2), gradient
        ).transpose(0, 1, 3, 2)

        # combine the two copies
        result = padded_array + padded_gradiated + flipped_gradiated
        result = np.where(padded_mask > 0, result / padded_mask, result)
        return flipped_array[:, :, pad_top : pad_top + latents.shape[2], :]
    else:
        raise ValueError("Invalid direction. Must be 'horizontal' or 'vertical'.")
