import unittest

from onnx_web.server.model_cache import ModelCache


class TestModelCache(unittest.TestCase):
  def test_drop_existing(self):
    cache = ModelCache(10)
    cache.clear()
    cache.set("foo", ("bar",), {})
    self.assertGreater(cache.size, 0)
    self.assertEqual(cache.drop("foo", ("bar",)), 1)

  def test_drop_missing(self):
    cache = ModelCache(10)
    cache.clear()
    cache.set("foo", ("bar",), {})
    self.assertGreater(cache.size, 0)
    self.assertEqual(cache.drop("foo", ("bin",)), 0)

  def test_get_existing(self):
    cache = ModelCache(10)
    cache.clear()
    value = {}
    cache.set("foo", ("bar",), value)
    self.assertGreater(cache.size, 0)
    self.assertIs(cache.get("foo", ("bar",)), value)

  def test_get_missing(self):
    cache = ModelCache(10)
    cache.clear()
    value = {}
    cache.set("foo", ("bar",), value)
    self.assertGreater(cache.size, 0)
    self.assertIs(cache.get("foo", ("bin",)), None)

  """
  def test_set_existing(self):
    cache = ModelCache(10)
    cache.clear()
    cache.set("foo", ("bar",), {
      "value": 1,
    })
    value = {
      "value": 2,
    }
    cache.set("foo", ("bar",), value)
    self.assertIs(cache.get("foo", ("bar",)), value)
  """

  def test_set_missing(self):
    cache = ModelCache(10)
    cache.clear()
    value = {}
    cache.set("foo", ("bar",), value)
    self.assertIs(cache.get("foo", ("bar",)), value)

  def test_set_zero(self):
    cache = ModelCache(0)
    cache.clear()
    value = {}
    cache.set("foo", ("bar",), value)
    self.assertEqual(cache.size, 0)
