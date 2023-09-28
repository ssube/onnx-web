import unittest

from PIL import Image

from onnx_web.image.mask_filter import (
    mask_filter_gaussian_multiply,
    mask_filter_gaussian_screen,
    mask_filter_none,
)


class MaskFilterNoneTests(unittest.TestCase):
  def test_basic(self):
    dims = (64, 64)
    mask = Image.new("RGB", dims)
    result = mask_filter_none(mask, dims, (0, 0))
    self.assertEqual(result.size, dims)


class MaskFilterGaussianMultiplyTests(unittest.TestCase):
  def test_basic(self):
    dims = (64, 64)
    mask = Image.new("RGB", dims)
    result = mask_filter_gaussian_multiply(mask, dims, (0, 0))
    self.assertEqual(result.size, dims)


class MaskFilterGaussianScreenTests(unittest.TestCase):
  def test_basic(self):
    dims = (64, 64)
    mask = Image.new("RGB", dims)
    result = mask_filter_gaussian_screen(mask, dims, (0, 0))
    self.assertEqual(result.size, dims)
