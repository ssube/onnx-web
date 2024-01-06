import unittest

from PIL import Image

from onnx_web.chain.blend_grid import BlendGridStage
from onnx_web.chain.result import ImageMetadata, StageResult
from onnx_web.params import ImageParams, Size


class BlendGridStageTests(unittest.TestCase):
    def test_stage(self):
        stage = BlendGridStage()
        sources = StageResult(
            images=[
                Image.new("RGB", (64, 64), "black"),
                Image.new("RGB", (64, 64), "white"),
                Image.new("RGB", (64, 64), "black"),
                Image.new("RGB", (64, 64), "white"),
            ],
            metadata=[
                ImageMetadata(
                    ImageParams("test", "txt2img", "ddim", "test", 1.0, 25, 1),
                    Size(64, 64),
                ),
            ]
            * 4,
        )
        result = stage.run(None, None, None, None, sources, height=2, width=2)
        result.validate()

        self.assertEqual(len(result), 5)
        self.assertEqual(result.as_images()[-1].getpixel((0, 0)), (0, 0, 0))
