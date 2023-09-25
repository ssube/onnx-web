import unittest

from onnx_web.convert.utils import DEFAULT_OPSET, ConversionContext, download_progress


class ConversionContextTests(unittest.TestCase):
    def test_from_environ(self):
      context = ConversionContext.from_environ()
      self.assertEqual(context.opset, DEFAULT_OPSET)

    def test_map_location(self):
      context = ConversionContext.from_environ()
      self.assertEqual(context.map_location.type, "cpu")


class DownloadProgressTests(unittest.TestCase):
   def test_download_example(self):
      path = download_progress([("https://example.com", "/tmp/example-dot-com")])
      self.assertEqual(path, "/tmp/example-dot-com")
