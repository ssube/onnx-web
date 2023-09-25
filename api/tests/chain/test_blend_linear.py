import unittest

from PIL import Image

from onnx_web.chain.blend_linear import BlendLinearStage


class BlendLinearStageTests(unittest.TestCase):
    def test_stage(self):
        stage = BlendLinearStage()
        sources = [
            Image.new("RGB", (64, 64), "black"),
        ]
        stage_source = Image.new("RGB", (64, 64), "white")
        result = stage.run(None, None, None, None, sources, alpha=0.5, stage_source=stage_source)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].getpixel((0,0)), (127, 127, 127))