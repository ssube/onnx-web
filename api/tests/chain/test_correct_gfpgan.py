import unittest

from onnx_web.chain.correct_gfpgan import CorrectGFPGANStage
from onnx_web.chain.result import StageResult
from onnx_web.params import HighresParams, UpscaleParams
from onnx_web.server.context import ServerContext
from onnx_web.server.hacks import apply_patches
from onnx_web.worker.context import WorkerContext
from tests.helpers import test_device, test_needs_models

TEST_MODEL_NAME = "correction-gfpgan-v1-3"
TEST_MODEL = f"../models/.cache/{TEST_MODEL_NAME}.pth"


class CorrectGFPGANStageTests(unittest.TestCase):
    @test_needs_models([TEST_MODEL])
    def test_empty(self):
        server = ServerContext(model_path="../models", output_path="../outputs")
        apply_patches(server)

        worker = WorkerContext(
            "test",
            test_device(),
            None,
            None,
            None,
            None,
            None,
            None,
            0,
            0.1,
        )
        stage = CorrectGFPGANStage()
        sources = StageResult.empty()
        result = stage.run(
            worker,
            server,
            None,
            None,
            sources,
            highres=HighresParams(False, 1, 0, 0),
            upscale=UpscaleParams(TEST_MODEL_NAME, TEST_MODEL_NAME),
        )
        result.validate()

        self.assertEqual(len(result), 0)
