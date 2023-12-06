import unittest

from onnx_web.chain.result import StageResult
from onnx_web.chain.source_s3 import SourceS3Stage
from onnx_web.params import HighresParams, Size, UpscaleParams


class SourceS3StageTests(unittest.TestCase):
    def test_empty(self):
        stage = SourceS3Stage()
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
            bucket="test",
            source_keys=[],
        )

        self.assertEqual(len(result), 0)
