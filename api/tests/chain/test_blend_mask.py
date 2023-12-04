import unittest

from PIL import Image

from onnx_web.chain.blend_mask import BlendMaskStage
from onnx_web.chain.result import StageResult
from onnx_web.params import HighresParams, UpscaleParams


class BlendMaskStageTests(unittest.TestCase):
    def test_empty(self):
        stage = BlendMaskStage()
        sources = StageResult.empty()
        result = stage.run(
            None,
            None,
            None,
            None,
            sources,
            highres=HighresParams(False, 1, 0, 0),
            upscale=UpscaleParams(""),
            stage_mask=Image.new("RGBA", (64, 64)),
        )

        self.assertEqual(len(result), 0)
