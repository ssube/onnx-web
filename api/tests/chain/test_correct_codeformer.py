import unittest

from onnx_web.chain.correct_codeformer import CorrectCodeformerStage
from onnx_web.params import DeviceParams, HighresParams, UpscaleParams
from onnx_web.server.context import ServerContext
from onnx_web.server.hacks import apply_patches
from onnx_web.worker.context import WorkerContext


class CorrectCodeformerStageTests(unittest.TestCase):
    def test_empty(self):
        """
        server = ServerContext()
        apply_patches(server)

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
        stage = CorrectCodeformerStage()
        sources = StageResult.empty()
        result = stage.run(worker, None, None, None, sources, highres=HighresParams(False,1, 0, 0), upscale=UpscaleParams(""))

        self.assertEqual(len(result), 0)
        """
        pass
