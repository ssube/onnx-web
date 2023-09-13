import unittest

from PIL import Image

from onnx_web.chain.tile import complete_tile


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
  def test_with_undersized(self):
    pass

  def test_with_oversized(self):
    pass

  def test_with_mixed(self):
    pass


class TestTileGrads(unittest.TestCase):
  def test_center_tile(self):
    pass

  def test_edge_tile(self):
    pass
