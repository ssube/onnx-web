import unittest

from PIL import Image

from onnx_web.chain.blend_img2img import BlendImg2ImgStage
from onnx_web.params import DeviceParams, ImageParams
from onnx_web.server.context import ServerContext
from onnx_web.worker.context import WorkerContext
from tests.helpers import TEST_MODEL_DIFFUSION_SD15, test_needs_models


class BlendImg2ImgStageTests(unittest.TestCase):
    @test_needs_models([TEST_MODEL_DIFFUSION_SD15])
    def test_stage(self):
        stage = BlendImg2ImgStage()
        params = ImageParams(
            TEST_MODEL_DIFFUSION_SD15,
            "txt2img",
            "euler-a",
            "an astronaut eating a hamburger",
            3.0,
            1,
            1,
        )
        server = ServerContext(model_path="../models", output_path="../outputs")
        worker = WorkerContext(
            "test",
            DeviceParams("cpu", "CPUProvider"),
            None,
            None,
            None,
            None,
            None,
            None,
            0,
        )
        sources = [
            Image.new("RGB", (64, 64), "black"),
        ]
        result = stage.run(worker, server, None, params, sources, strength=0.5, steps=1)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].getpixel((0, 0)), (127, 127, 127))
