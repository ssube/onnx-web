import unittest

from onnx_web.utils import (
    get_and_clamp_float,
    get_and_clamp_int,
    get_boolean,
    get_from_list,
    get_from_map,
    get_list,
    get_not_empty,
    split_list,
)


class TestUtils(unittest.TestCase):
    def test_split_list_empty(self):
        self.assertEqual(split_list(""), [])
        self.assertEqual(split_list(" "), [])
        self.assertEqual(split_list(" , "), [])

    def test_split_list_single(self):
        self.assertEqual(split_list("a"), ["a"])
        self.assertEqual(split_list(" a "), ["a"])
        self.assertEqual(split_list(" a, "), ["a"])
        self.assertEqual(split_list(" a , "), ["a"])

    def test_split_list_multiple(self):
        self.assertEqual(split_list("a,b"), ["a", "b"])
        self.assertEqual(split_list(" a , b "), ["a", "b"])
        self.assertEqual(split_list(" a, b "), ["a", "b"])
        self.assertEqual(split_list(" a ,b "), ["a", "b"])

    def test_get_boolean_empty(self):
        self.assertFalse(get_boolean({}, "key", False))
        self.assertTrue(get_boolean({}, "key", True))

    def test_get_boolean_true(self):
        self.assertTrue(get_boolean({"key": True}, "key", False))
        self.assertTrue(get_boolean({"key": True}, "key", True))

    def test_get_boolean_false(self):
        self.assertFalse(get_boolean({"key": False}, "key", False))
        self.assertFalse(get_boolean({"key": False}, "key", True))

    def test_get_list_empty(self):
        self.assertEqual(get_list({}, "key", ""), [])
        self.assertEqual(get_list({}, "key", "a"), ["a"])

    def test_get_list_exists(self):
        self.assertEqual(get_list({"key": "a,b"}, "key", ""), ["a", "b"])
        self.assertEqual(get_list({"key": "a,b"}, "key", "c"), ["a", "b"])

    def test_get_and_clamp_float_empty(self):
        self.assertEqual(get_and_clamp_float({}, "key", 0.0, 1.0), 0.0)
        self.assertEqual(get_and_clamp_float({}, "key", 1.0, 1.0), 1.0)

    def test_get_and_clamp_float_clamped(self):
        self.assertEqual(get_and_clamp_float({"key": -1.0}, "key", 0.0, 1.0), 0.0)
        self.assertEqual(get_and_clamp_float({"key": 2.0}, "key", 0.0, 1.0), 1.0)

    def test_get_and_clamp_float_normal(self):
        self.assertEqual(get_and_clamp_float({"key": 0.5}, "key", 0.0, 1.0), 0.5)

    def test_get_and_clamp_int_empty(self):
        self.assertEqual(get_and_clamp_int({}, "key", 0, 1), 1)
        self.assertEqual(get_and_clamp_int({}, "key", 1, 1), 1)

    def test_get_and_clamp_int_clamped(self):
        self.assertEqual(get_and_clamp_int({"key": 0}, "key", 1, 1), 1)
        self.assertEqual(get_and_clamp_int({"key": 2}, "key", 1, 1), 1)

    def test_get_and_clamp_int_normal(self):
        self.assertEqual(get_and_clamp_int({"key": 1}, "key", 0, 1), 1)

    def test_get_from_list_empty(self):
        self.assertEqual(get_from_list({}, "key", ["a", "b"]), "a")
        self.assertEqual(get_from_list({}, "key", ["a", "b"], "a"), "a")

    def test_get_from_list_exists(self):
        self.assertEqual(get_from_list({"key": "a"}, "key", ["a", "b"]), "a")
        self.assertEqual(get_from_list({"key": "b"}, "key", ["a", "b"]), "b")

    def test_get_from_list_invalid(self):
        self.assertEqual(get_from_list({"key": "c"}, "key", ["a", "b"]), "a")

    def test_get_from_map_empty(self):
        self.assertEqual(get_from_map({}, "key", {"a": 1, "b": 2}, "a"), 1)
        self.assertEqual(get_from_map({}, "key", {"a": 1, "b": 2}, "b"), 2)

    def test_get_from_map_exists(self):
        self.assertEqual(get_from_map({"key": "a"}, "key", {"a": 1, "b": 2}, "a"), 1)
        self.assertEqual(get_from_map({"key": "b"}, "key", {"a": 1, "b": 2}, "a"), 2)

    def test_get_not_empty_empty(self):
        self.assertEqual(get_not_empty({}, "key", "a"), "a")
        self.assertEqual(get_not_empty({"key": ""}, "key", "a"), "a")

    def test_get_not_empty_exists(self):
        self.assertEqual(get_not_empty({"key": "b"}, "key", "a"), "b")
