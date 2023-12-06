import unittest

from onnx_web.models.meta import NetworkModel


class NetworkModelTests(unittest.TestCase):
    def test_json(self):
        model = NetworkModel("test", "inversion")
        json = model.tojson()

        self.assertIn("name", json)
        self.assertIn("type", json)
