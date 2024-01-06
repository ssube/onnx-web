import unittest

from PIL import Image

from onnx_web.chain.blend_linear import BlendLinearStage
from onnx_web.chain.result import ImageMetadata, StageResult
from onnx_web.params import ImageParams, Size


class BlendLinearStageTests(unittest.TestCase):
    def test_stage(self):
        stage = BlendLinearStage()
        sources = StageResult(
            images=[
                Image.new("RGB", (64, 64), "black"),
            ],
            metadata=[
                ImageMetadata(
                    ImageParams("test", "txt2img", "ddim", "test", 1.0, 25, 1),
                    Size(64, 64),
                ),
            ],
        )
        stage_source = Image.new("RGB", (64, 64), "white")
        result = stage.run(
            None, None, None, None, sources, alpha=0.5, stage_source=stage_source
        )
        result.validate()

        self.assertEqual(len(result), 1)
        self.assertEqual(result.as_images()[0].getpixel((0, 0)), (127, 127, 127))
