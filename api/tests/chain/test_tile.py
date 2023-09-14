import unittest

from PIL import Image

from onnx_web.chain.tile import complete_tile, get_tile_grads, needs_tile
from onnx_web.params import Size


class TestCompleteTile(unittest.TestCase):
  def test_with_complete_tile(self):
    partial = Image.new("RGB", (64, 64))
    output = complete_tile(partial, 64)

    self.assertEqual(output.size, (64, 64))

  def test_with_partial_tile(self):
    partial = Image.new("RGB", (64, 32))
    output = complete_tile(partial, 64)

    self.assertEqual(output.size, (64, 64))

  def test_with_nothing(self):
    output = complete_tile(None, 64)

    self.assertIsNone(output)


class TestNeedsTile(unittest.TestCase):
  def test_with_undersized_source(self):
    small = Image.new("RGB", (32, 32))

    self.assertFalse(needs_tile(64, 64, source=small))

  def test_with_oversized_source(self):
    large = Image.new("RGB", (64, 64))

    self.assertTrue(needs_tile(32, 32, source=large))

  def test_with_undersized_size(self):
    small = Size(32, 32)

    self.assertFalse(needs_tile(64, 64, size=small))

  def test_with_oversized_source(self):
    large = Size(64, 64)

    self.assertTrue(needs_tile(32, 32, size=large))

  def test_with_nothing(self):
    self.assertFalse(needs_tile(32, 32))


class TestTileGrads(unittest.TestCase):
  def test_center_tile(self):
    grad_x, grad_y = get_tile_grads(32, 32, 8, 64, 64)

    self.assertEqual(grad_x, [0, 1, 1, 0])
    self.assertEqual(grad_y, [0, 1, 1, 0])

  def test_vertical_edge_tile(self):
    grad_x, grad_y = get_tile_grads(32, 0, 8, 64, 8)

    self.assertEqual(grad_x, [0, 1, 1, 0])
    self.assertEqual(grad_y, [1, 1, 1, 1])


  def test_horizontal_edge_tile(self):
    grad_x, grad_y = get_tile_grads(0, 32, 8, 8, 64)

    self.assertEqual(grad_x, [1, 1, 1, 1])
    self.assertEqual(grad_y, [0, 1, 1, 0])
