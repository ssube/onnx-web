import unittest

from onnx_web.chain.result import StageResult
from onnx_web.chain.source_noise import SourceNoiseStage
from onnx_web.image.noise_source import noise_source_fill_edge
from onnx_web.params import HighresParams, Size, UpscaleParams


class SourceNoiseStageTests(unittest.TestCase):
    def test_empty(self):
        stage = SourceNoiseStage()
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
            noise_source=noise_source_fill_edge,
        )
        result.validate()

        self.assertEqual(len(result), 0)
