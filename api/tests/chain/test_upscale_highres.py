import unittest

from onnx_web.chain.result import StageResult
from onnx_web.chain.upscale_highres import UpscaleHighresStage
from onnx_web.params import HighresParams, UpscaleParams


class UpscaleHighresStageTests(unittest.TestCase):
    def test_empty(self):
        stage = UpscaleHighresStage()
        sources = StageResult.empty()
        result = stage.run(None, None, None, None, sources, highres=HighresParams(False,1, 0, 0), upscale=UpscaleParams(""))

        self.assertEqual(len(result), 0)
