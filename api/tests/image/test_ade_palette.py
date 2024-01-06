import unittest

from onnx_web.image.ade_palette import ade_palette


class TestADEPalette(unittest.TestCase):
    def test_palette_length(self):
        palette = ade_palette()
        self.assertEqual(len(palette), 150, "Palette length should be 150")
