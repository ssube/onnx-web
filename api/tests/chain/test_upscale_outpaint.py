import unittest

from PIL import Image

from onnx_web.chain.result import StageResult
from onnx_web.chain.upscale_outpaint import UpscaleOutpaintStage
from onnx_web.params import Border, HighresParams, ImageParams, UpscaleParams
from onnx_web.server.context import ServerContext
from onnx_web.worker.context import WorkerContext
from tests.helpers import test_device, test_needs_models


class UpscaleOutpaintStageTests(unittest.TestCase):
    @test_needs_models(["../models/stable-diffusion-onnx-v1-inpainting"])
    def test_empty(self):
        stage = UpscaleOutpaintStage()
        sources = StageResult.empty()
        result = stage.run(
            WorkerContext(
                "test",
                test_device(),
                None,
                None,
                None,
                None,
                None,
                None,
                3,
                0.1,
            ),
            ServerContext(),
            None,
            ImageParams(
                "../models/stable-diffusion-onnx-v1-inpainting",
                "inpaint",
                "euler",
                "test",
                5.0,
                1,
                1,
            ),
            sources,
            highres=HighresParams(False, 1, 0, 0),
            upscale=UpscaleParams("stable-diffusion-onnx-v1-inpainting"),
            border=Border.even(0),
            dims=(),
            tile_mask=Image.new("RGB", (64, 64)),
        )
        result.validate()

        self.assertEqual(len(result), 0)
