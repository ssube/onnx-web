import unittest

from onnx_web.chain.result import StageResult
from onnx_web.chain.upscale_resrgan import UpscaleRealESRGANStage
from onnx_web.params import HighresParams, StageParams, UpscaleParams
from onnx_web.server.context import ServerContext
from onnx_web.worker.context import WorkerContext
from tests.helpers import test_device, test_needs_onnx_models

TEST_MODEL = "../models/upscaling-real-esrgan-x4-v3"


class UpscaleRealESRGANStageTests(unittest.TestCase):
    @test_needs_onnx_models([TEST_MODEL])
    def test_empty(self):
        stage = UpscaleRealESRGANStage()
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
            ServerContext(model_path="../models"),
            StageParams(),
            None,
            sources,
            highres=HighresParams(False, 1, 0, 0),
            upscale=UpscaleParams("upscaling-real-esrgan-x4-v3"),
        )

        self.assertEqual(len(result), 0)
