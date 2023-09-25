import unittest

from PIL import Image

from onnx_web.chain.blend_grid import BlendGridStage
from onnx_web.chain.blend_linear import BlendLinearStage


class BlendGridStageTests(unittest.TestCase):
    def test_stage(self):
        stage = BlendGridStage()
        sources = [
            Image.new("RGB", (64, 64), "black"),
            Image.new("RGB", (64, 64), "white"),
            Image.new("RGB", (64, 64), "black"),
            Image.new("RGB", (64, 64), "white"),
        ]
        result = stage.run(None, None, None, None, sources, height=2, width=2)

        self.assertEqual(len(result), 5)
        self.assertEqual(result[-1].getpixel((0,0)), (0, 0, 0))