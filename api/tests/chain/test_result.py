import unittest

from onnx_web.chain.result import ImageMetadata


class ImageMetadataTests(unittest.TestCase):
    def test_image_metadata(self):
        pass

    def test_from_exif_normal(self):
        exif_data = """test prompt
Negative prompt: negative prompt
Sampler: ddim, CFG scale: 4.0, Steps: 30, Seed: 5
"""

        metadata = ImageMetadata.from_exif(exif_data)
        self.assertEqual(metadata.params.prompt, "test prompt")
        self.assertEqual(metadata.params.negative_prompt, "negative prompt")
        self.assertEqual(metadata.params.scheduler, "ddim")
        self.assertEqual(metadata.params.cfg, 4.0)
        self.assertEqual(metadata.params.steps, 30)
        self.assertEqual(metadata.params.seed, 5)

    def test_from_exif_split(self):
        exif_data = """test prompt
Negative prompt: negative prompt
Sampler: ddim,
CFG scale: 4.0,
Steps: 30, Seed: 5
"""

        metadata = ImageMetadata.from_exif(exif_data)
        self.assertEqual(metadata.params.prompt, "test prompt")
        self.assertEqual(metadata.params.negative_prompt, "negative prompt")
        self.assertEqual(metadata.params.scheduler, "ddim")
        self.assertEqual(metadata.params.cfg, 4.0)
        self.assertEqual(metadata.params.steps, 30)
        self.assertEqual(metadata.params.seed, 5)


class StageResultTests(unittest.TestCase):
    def test_stage_result(self):
        pass
