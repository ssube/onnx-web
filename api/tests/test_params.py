import unittest

from onnx_web.params import Border, Size

class BorderTests(unittest.TestCase):
    def test_json(self):
        border = Border.even(0)
        json = border.tojson()

        self.assertIn("left", json)
        self.assertIn("right", json)
        self.assertIn("top", json)
        self.assertIn("bottom", json)

    def test_str(self):
        border = Border.even(10)
        bstr = str(border)

        self.assertEqual("(10, 10, 10, 10)", bstr)

    def test_uneven(self):
        border = Border(1, 2, 3, 4)

        self.assertEqual("(1, 2, 3, 4)", str(border))

    def test_args(self):
        pass


class SizeTests(unittest.TestCase):
    def test_iter(self):
        size = Size(1, 2)

        self.assertEqual(list(size), [1, 2])

    def test_str(self):
        pass

    def test_border(self):
        pass

    def test_json(self):
        pass

    def test_args(self):
        pass


class DeviceParamsTests(unittest.TestCase):
    def test_str(self):
        pass

    def test_provider(self):
        pass

    def test_options_optimizations(self):
        pass

    def test_options_cache(self):
        pass

    def test_torch_cuda(self):
        pass

    def test_torch_rocm(self):
        pass


class ImageParamsTests(unittest.TestCase):
    def test_json(self):
        pass

    def test_args(self):
        pass


class StageParamsTests(unittest.TestCase):
    def test_init(self):
        pass


class UpscaleParamsTests(unittest.TestCase):
    def test_rescale(self):
        pass

    def test_resize(self):
        pass

    def test_json(self):
        pass

    def test_args(self):
        pass


class HighresParamsTests(unittest.TestCase):
    def test_resize(self):
        pass

    def test_json(self):
        pass
