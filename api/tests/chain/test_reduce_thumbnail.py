import unittest

from PIL import Image

from onnx_web.chain.reduce_thumbnail import ReduceThumbnailStage
from onnx_web.chain.result import StageResult
from onnx_web.params import HighresParams, Size, UpscaleParams


class ReduceThumbnailStageTests(unittest.TestCase):
    def test_empty(self):
        stage_source = Image.new("RGB", (64, 64))
        stage = ReduceThumbnailStage()
        sources = StageResult.empty()
        result = stage.run(
            None,
            None,
            None,
            None,
            sources,
            highres=HighresParams(False, 1, 0, 0),
            upscale=UpscaleParams(""),
            origin=Size(0, 0),
            size=Size(128, 128),
            stage_source=stage_source,
        )

        self.assertEqual(len(result), 0)
