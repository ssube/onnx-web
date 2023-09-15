import unittest

from onnx_web.chain.source_url import SourceURLStage
from onnx_web.params import HighresParams, Size, UpscaleParams


class SourceURLStageTests(unittest.TestCase):
    def test_empty(self):
        stage = SourceURLStage()
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
            source_urls=[],
        )

        self.assertEqual(len(result), 0)
