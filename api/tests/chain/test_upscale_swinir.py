import unittest

from onnx_web.chain.result import StageResult
from onnx_web.chain.upscale_swinir import UpscaleSwinIRStage
from onnx_web.params import HighresParams, UpscaleParams
from onnx_web.server.context import ServerContext
from onnx_web.worker.context import WorkerContext
from tests.helpers import test_device, test_needs_onnx_models

TEST_MODEL = "../models/upscaling-swinir-real-large-x4"


class UpscaleSwinIRStageTests(unittest.TestCase):
    @test_needs_onnx_models([TEST_MODEL])
    def test_empty(self):
        stage = UpscaleSwinIRStage()
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
            None,
            sources,
            highres=HighresParams(False, 1, 0, 0),
            upscale=UpscaleParams(TEST_MODEL),
        )
        result.validate()

        self.assertEqual(len(result), 0)
