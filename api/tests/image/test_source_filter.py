import unittest

from PIL import Image

from onnx_web.image.source_filter import (
    source_filter_gaussian,
    source_filter_noise,
    source_filter_none,
)
from onnx_web.server.context import ServerContext


class SourceFilterNoneTests(unittest.TestCase):
    def test_basic(self):
        dims = (64, 64)
        server = ServerContext()
        source = Image.new("RGB", dims)
        result = source_filter_none(server, source)
        self.assertEqual(result.size, dims)


class SourceFilterGaussianTests(unittest.TestCase):
    def test_basic(self):
        dims = (64, 64)
        server = ServerContext()
        source = Image.new("RGB", dims)
        result = source_filter_gaussian(server, source)
        self.assertEqual(result.size, dims)


class SourceFilterNoiseTests(unittest.TestCase):
    def test_basic(self):
        dims = (64, 64)
        server = ServerContext()
        source = Image.new("RGB", dims)
        result = source_filter_noise(server, source)
        self.assertEqual(result.size, dims)
