import unittest
from unittest.mock import MagicMock

import numpy as np
import torch
from diffusers.schedulers.scheduling_utils import SchedulerOutput
from numpy.testing import assert_array_equal

from onnx_web.diffusers.patches.scheduler import (
    SchedulerPatch,
    linear_gradient,
    mirror_latents,
)


class SchedulerPatchTests(unittest.TestCase):
    def test_scheduler_step(self):
        wrapped_scheduler = MagicMock()
        wrapped_scheduler.step.return_value = SchedulerOutput(None)
        scheduler = SchedulerPatch(None, wrapped_scheduler, None)
        model_output = torch.FloatTensor([1.0, 2.0, 3.0])
        timestep = torch.Tensor([0.1])
        sample = torch.FloatTensor([0.5, 0.6, 0.7])
        output = scheduler.step(model_output, timestep, sample)
        self.assertIsInstance(output, SchedulerOutput)

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
        gradient = linear_gradient(white_point, black_point, center_line)
        mirrored_latents = mirror_latents(latents, gradient, center_line, direction)
        assert_array_equal(mirrored_latents, latents)

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
        gradient = linear_gradient(white_point, black_point, center_line)
        mirrored_latents = mirror_latents(latents, gradient, center_line, direction)
        assert_array_equal(
            mirrored_latents,
            [
                [
                    [[0, 0, 0], [0, 0, 0], [10, 11, 12], [7, 8, 9]],
                ]
            ],
        )
