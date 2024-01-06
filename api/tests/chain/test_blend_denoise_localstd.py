import unittest

from PIL import Image

from onnx_web.chain.blend_denoise_localstd import BlendDenoiseLocalStdStage
from onnx_web.chain.result import ImageMetadata, StageResult
from onnx_web.params import ImageParams, Size


class TestBlendDenoiseLocalStdStage(unittest.TestCase):
    def test_run(self):
        # Create a dummy image
        image = Image.new("RGB", (64, 64), color="white")

        # Create a dummy StageResult object
        sources = StageResult.from_images(
            [image],
            metadata=[
                ImageMetadata(
                    ImageParams("test", "txt2img", "ddim", "test", 5.0, 25, 0),
                    Size(64, 64),
                )
            ],
        )

        # Create an instance of BlendDenoiseLocalStdStage
        stage = BlendDenoiseLocalStdStage()

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
        self.assertEqual(result.size(), Size(64, 64))

        # Assert that the metadata is preserved
        self.assertEqual(result.metadata, sources.metadata)
