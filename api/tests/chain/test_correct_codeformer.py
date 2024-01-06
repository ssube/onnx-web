import unittest

from onnx_web.chain.correct_codeformer import CorrectCodeformerStage
from onnx_web.chain.result import StageResult
from onnx_web.params import HighresParams, UpscaleParams
from onnx_web.server.context import ServerContext
from onnx_web.server.hacks import apply_patches
from onnx_web.worker.context import WorkerContext
from tests.helpers import (
    TEST_MODEL_CORRECTION_CODEFORMER,
    test_device,
    test_needs_models,
)


class CorrectCodeformerStageTests(unittest.TestCase):
    @test_needs_models([TEST_MODEL_CORRECTION_CODEFORMER])
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
        stage = CorrectCodeformerStage()
        sources = StageResult.empty()
        result = stage.run(
            worker,
            server,
            None,
            None,
            sources,
            highres=HighresParams(False, 1, 0, 0),
            upscale=UpscaleParams(""),
        )
        result.validate()

        self.assertEqual(len(result), 0)
