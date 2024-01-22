import unittest

import numpy as np
import torch
from diffusers.schedulers.scheduling_ddim import DDIMSchedulerOutput

from onnx_web.diffusers.patches.scheduler import SchedulerPatch, mirror_latents


class SchedulerPatchTests(unittest.TestCase):
    def test_scheduler_step(self):
        scheduler = SchedulerPatch(None)
        model_output = torch.FloatTensor([1.0, 2.0, 3.0])
        timestep = torch.Tensor([0.1])
        sample = torch.FloatTensor([0.5, 0.6, 0.7])
        output = scheduler.step(model_output, timestep, sample)
        assert isinstance(output, DDIMSchedulerOutput)

    def test_mirror_latents_horizontal(self):
        latents = np.array(
            [  # batch
                [  # channels
                    [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
                ],
            ]
        )
        white_point = 0
        black_point = 1
        center_line = 2
        direction = "horizontal"
        mirrored_latents = mirror_latents(
            latents, white_point, black_point, center_line, direction
        )
        assert np.array_equal(mirrored_latents, latents)

    def test_mirror_latents_vertical(self):
        latents = np.array(
            [  # batch
                [  # channels
                    [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
                ],
            ]
        )
        white_point = 0
        black_point = 1
        center_line = 3
        direction = "vertical"
        mirrored_latents = mirror_latents(
            latents, white_point, black_point, center_line, direction
        )
        assert np.array_equal(
            mirrored_latents,
            [
                [
                    [[0, 0, 0], [0, 0, 0], [10, 11, 12], [7, 8, 9]],
                ]
            ],
        )
