import unittest

from PIL import Image

from onnx_web.chain.blend_denoise_fastnlmeans import BlendDenoiseFastNLMeansStage
from onnx_web.chain.result import ImageMetadata, StageResult
from tests.helpers import test_params, test_size


class TestBlendDenoiseFastNLMeansStage(unittest.TestCase):
    def test_run(self):
        # Create a dummy image
        size = test_size()
        image = Image.new("RGB", (size.width, size.height), color="white")

        # Create a dummy StageResult object
        sources = StageResult.from_images(
            [image],
            metadata=[
                ImageMetadata(
                    test_params(),
                    size,
                )
            ],
        )

        # Create an instance of BlendDenoiseLocalStdStage
        stage = BlendDenoiseFastNLMeansStage()

        # Call the run method with dummy parameters
        result = stage.run(
            _worker=None,
            _server=None,
            _stage=None,
            _params=None,
            sources=sources,
            strength=5,
            range=4,
            stage_source=None,
            callback=None,
        )

        # Assert that the result is an instance of StageResult
        self.assertIsInstance(result, StageResult)

        # Assert that the result contains the denoised image
        self.assertEqual(len(result), 1)
        self.assertEqual(result.size(), size)

        # Assert that the metadata is preserved
        self.assertEqual(result.metadata, sources.metadata)
