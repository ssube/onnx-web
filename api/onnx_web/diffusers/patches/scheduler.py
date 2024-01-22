from typing import Any, Literal

import numpy as np
import torch
from diffusers.schedulers.scheduling_utils import SchedulerOutput
from torch import FloatTensor, Tensor


class SchedulerPatch:
    wrapped: Any

    def __init__(self, scheduler):
        self.wrapped = scheduler

    def __getattr__(self, attr):
        return getattr(self.wrapped, attr)

    def step(
        self, model_output: FloatTensor, timestep: Tensor, sample: FloatTensor
    ) -> SchedulerOutput:
        result = self.wrapped.step(model_output, timestep, sample)

        white_point = result.prev_sample.shape[2] // 8
        black_point = result.prev_sample.shape[2] // 4
        center_line = result.prev_sample.shape[2] // 2
        direction = "horizontal"

        mirrored_latents = mirror_latents(
            result.prev_sample.numpy(), white_point, black_point, center_line, direction
        )

        return SchedulerOutput(
            prev_sample=torch.from_numpy(mirrored_latents),
        )


def mirror_latents(
    latents: np.ndarray,
    white_point: int,
    black_point: int,
    center_line: int,
    direction: Literal["horizontal", "vertical"],
) -> np.ndarray:
    gradient = np.linspace(1, 0, black_point - white_point).astype(np.float32)
    gradient = np.pad(
        gradient, (white_point, center_line - black_point), mode="constant"
    )
    gradient = np.reshape([gradient, np.flip(gradient)], -1)
    gradient = np.expand_dims(gradient, (0, 1, 2))

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
        padded_mask += np.multiply(np.ones_like(padded_array), gradient)

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
        padded_mask += np.multiply(
            np.ones_like(padded_array).transpose(0, 1, 3, 2), gradient
        ).transpose(0, 1, 3, 2)

        # combine the two copies
        result = padded_array + padded_gradiated + flipped_gradiated
        result = np.where(padded_mask > 0, result / padded_mask, result)
        return flipped_array[:, :, pad_top : pad_top + latents.shape[2], :]
    else:
        raise ValueError("Invalid direction. Must be 'horizontal' or 'vertical'.")
