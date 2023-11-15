import unittest

from PIL import Image

from onnx_web.image.utils import expand_image
from onnx_web.params import Border


class ExpandImageTests(unittest.TestCase):
  def test_expand(self):
    result = expand_image(
      Image.new("RGB", (8, 8)),
      Image.new("RGB", (8, 8), "white"),
      Border.even(4),
    )
    self.assertEqual(result[0].size, (16, 16))

  def test_masked(self):
    result = expand_image(
      Image.new("RGB", (8, 8), "red"),
      Image.new("RGB", (8, 8), "white"),
      Border.even(4),
    )
    self.assertEqual(result[0].getpixel((8, 8)), (255, 0, 0))
