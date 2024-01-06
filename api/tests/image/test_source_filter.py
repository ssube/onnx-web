import unittest
from os import path

import numpy as np
from PIL import Image

from onnx_web.image.source_filter import (
    filter_model_path,
    pil_to_cv2,
    source_filter_canny,
    source_filter_depth,
    source_filter_face,
    source_filter_gaussian,
    source_filter_hed,
    source_filter_mlsd,
    source_filter_noise,
    source_filter_none,
    source_filter_normal,
    source_filter_openpose,
    source_filter_scribble,
    source_filter_segment,
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


class PILToCV2Tests(unittest.TestCase):
    def test_conversion(self):
        dims = (64, 64)
        source = Image.new("RGB", dims)
        result = pil_to_cv2(source)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (dims[1], dims[0], 3))
        self.assertEqual(result.dtype, np.uint8)


class FilterModelPathTests(unittest.TestCase):
    def test_filter_model_path(self):
        server = ServerContext()
        filter_name = "gaussian"
        expected_path = path.join(server.model_path, "filter", filter_name)
        result = filter_model_path(server, filter_name)
        self.assertEqual(result, expected_path)


class SourceFilterFaceTests(unittest.TestCase):  # Added new test class
    def test_basic(self):
        dims = (64, 64)
        server = ServerContext()
        source = Image.new("RGB", dims)
        result = source_filter_face(server, source)
        self.assertEqual(result.size, dims)


class SourceFilterSegmentTests(
    unittest.TestCase
):  # Added SourceFilterSegmentTests class
    def test_basic(self):
        dims = (64, 64)
        server = ServerContext()
        source = Image.new("RGB", dims)
        result = source_filter_segment(server, source)
        self.assertEqual(result.size, dims)


class SourceFilterMLSDTests(unittest.TestCase):  # Added SourceFilterMLSDTests class
    def test_basic(self):
        dims = (64, 64)
        server = ServerContext()
        source = Image.new("RGB", dims)
        result = source_filter_mlsd(server, source)
        self.assertEqual(result.size, (512, 512))


class SourceFilterNormalTests(unittest.TestCase):  # Added SourceFilterNormalTests class
    def test_basic(self):
        dims = (64, 64)
        server = ServerContext()
        source = Image.new("RGB", dims)
        result = source_filter_normal(server, source)

        # normal will resize inputs to 384x384
        self.assertEqual(result.size, (384, 384))


class SourceFilterHEDTests(unittest.TestCase):
    def test_basic(self):
        dims = (64, 64)
        server = ServerContext()
        source = Image.new("RGB", dims)
        result = source_filter_hed(server, source)
        self.assertEqual(result.size, (512, 512))


class SourceFilterScribbleTests(
    unittest.TestCase
):  # Added SourceFilterScribbleTests class
    def test_basic(self):
        dims = (64, 64)
        server = ServerContext()
        source = Image.new("RGB", dims)
        result = source_filter_scribble(server, source)

        # scribble will resize inputs to 512x512
        self.assertEqual(result.size, (512, 512))


class SourceFilterDepthTests(
    unittest.TestCase
):  # Added SourceFilterScribbleTests class
    def test_basic(self):
        dims = (64, 64)
        server = ServerContext()
        source = Image.new("RGB", dims)
        result = source_filter_depth(server, source)
        self.assertEqual(result.size, dims)


class SourceFilterCannyTests(
    unittest.TestCase
):  # Added SourceFilterScribbleTests class
    def test_basic(self):
        dims = (64, 64)
        server = ServerContext()
        source = Image.new("RGB", dims)
        result = source_filter_canny(server, source)
        self.assertEqual(result.size, dims)


class SourceFilterOpenPoseTests(
    unittest.TestCase
):  # Added SourceFilterScribbleTests class
    def test_basic(self):
        dims = (64, 64)
        server = ServerContext()
        source = Image.new("RGB", dims)
        result = source_filter_openpose(server, source)

        # openpose will resize inputs to 512x512
        self.assertEqual(result.size, (512, 512))
