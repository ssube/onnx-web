import unittest

from onnx_web.chain.reduce_crop import ReduceCropStage
from onnx_web.params import HighresParams, Size, UpscaleParams


class ReduceCropStageTests(unittest.TestCase):
    def test_empty(self):
        stage = ReduceCropStage()
        sources = []
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
        )

        self.assertEqual(len(result), 0)
