import unittest

from onnx_web.convert.utils import (
    DEFAULT_OPSET,
    ConversionContext,
    download_progress,
    tuple_to_correction,
    tuple_to_diffusion,
    tuple_to_source,
    tuple_to_upscaling,
)


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


class TupleToSourceTests(unittest.TestCase):
   def test_basic_tuple(self):
      source = tuple_to_source(("foo", "bar"))
      self.assertEqual(source["name"], "foo")
      self.assertEqual(source["source"], "bar")

   def test_basic_list(self):
      source = tuple_to_source(["foo", "bar"])
      self.assertEqual(source["name"], "foo")
      self.assertEqual(source["source"], "bar")

   def test_basic_dict(self):
      source = tuple_to_source(["foo", "bar"])
      source["bin"] = "bin"

      # make sure this is returned as-is with extra fields
      second = tuple_to_source(source)

      self.assertEqual(source, second)
      self.assertIn("bin", second)


class TupleToCorrectionTests(unittest.TestCase):
   def test_basic_tuple(self):
      source = tuple_to_correction(("foo", "bar"))
      self.assertEqual(source["name"], "foo")
      self.assertEqual(source["source"], "bar")

   def test_basic_list(self):
      source = tuple_to_correction(["foo", "bar"])
      self.assertEqual(source["name"], "foo")
      self.assertEqual(source["source"], "bar")

   def test_basic_dict(self):
      source = tuple_to_correction(["foo", "bar"])
      source["bin"] = "bin"

      # make sure this is returned with extra fields
      second = tuple_to_source(source)

      self.assertEqual(source, second)
      self.assertIn("bin", second)

   def test_scale_tuple(self):
      source = tuple_to_correction(["foo", "bar", 2])
      self.assertEqual(source["name"], "foo")
      self.assertEqual(source["source"], "bar")

   def test_half_tuple(self):
      source = tuple_to_correction(["foo", "bar", True])
      self.assertEqual(source["name"], "foo")
      self.assertEqual(source["source"], "bar")

   def test_opset_tuple(self):
      source = tuple_to_correction(["foo", "bar", 14])
      self.assertEqual(source["name"], "foo")
      self.assertEqual(source["source"], "bar")

   def test_all_tuple(self):
      source = tuple_to_correction(["foo", "bar", 2, True, 14])
      self.assertEqual(source["name"], "foo")
      self.assertEqual(source["source"], "bar")
      self.assertEqual(source["scale"], 2)
      self.assertEqual(source["half"], True)
      self.assertEqual(source["opset"], 14)


class TupleToDiffusionTests(unittest.TestCase):
   def test_basic_tuple(self):
      source = tuple_to_diffusion(("foo", "bar"))
      self.assertEqual(source["name"], "foo")
      self.assertEqual(source["source"], "bar")

   def test_basic_list(self):
      source = tuple_to_diffusion(["foo", "bar"])
      self.assertEqual(source["name"], "foo")
      self.assertEqual(source["source"], "bar")

   def test_basic_dict(self):
      source = tuple_to_diffusion(["foo", "bar"])
      source["bin"] = "bin"

      # make sure this is returned with extra fields
      second = tuple_to_diffusion(source)

      self.assertEqual(source, second)
      self.assertIn("bin", second)

   def test_single_vae_tuple(self):
      source = tuple_to_diffusion(["foo", "bar", True])
      self.assertEqual(source["name"], "foo")
      self.assertEqual(source["source"], "bar")

   def test_half_tuple(self):
      source = tuple_to_diffusion(["foo", "bar", True])
      self.assertEqual(source["name"], "foo")
      self.assertEqual(source["source"], "bar")

   def test_opset_tuple(self):
      source = tuple_to_diffusion(["foo", "bar", 14])
      self.assertEqual(source["name"], "foo")
      self.assertEqual(source["source"], "bar")

   def test_all_tuple(self):
      source = tuple_to_diffusion(["foo", "bar", True, True, 14])
      self.assertEqual(source["name"], "foo")
      self.assertEqual(source["source"], "bar")
      self.assertEqual(source["single_vae"], True)
      self.assertEqual(source["half"], True)
      self.assertEqual(source["opset"], 14)


class TupleToUpscalingTests(unittest.TestCase):
   def test_basic_tuple(self):
      source = tuple_to_upscaling(("foo", "bar"))
      self.assertEqual(source["name"], "foo")
      self.assertEqual(source["source"], "bar")

   def test_basic_list(self):
      source = tuple_to_upscaling(["foo", "bar"])
      self.assertEqual(source["name"], "foo")
      self.assertEqual(source["source"], "bar")

   def test_basic_dict(self):
      source = tuple_to_upscaling(["foo", "bar"])
      source["bin"] = "bin"

      # make sure this is returned with extra fields
      second = tuple_to_source(source)

      self.assertEqual(source, second)
      self.assertIn("bin", second)

   def test_scale_tuple(self):
      source = tuple_to_upscaling(["foo", "bar", 2])
      self.assertEqual(source["name"], "foo")
      self.assertEqual(source["source"], "bar")

   def test_half_tuple(self):
      source = tuple_to_upscaling(["foo", "bar", True])
      self.assertEqual(source["name"], "foo")
      self.assertEqual(source["source"], "bar")

   def test_opset_tuple(self):
      source = tuple_to_upscaling(["foo", "bar", 14])
      self.assertEqual(source["name"], "foo")
      self.assertEqual(source["source"], "bar")

   def test_all_tuple(self):
      source = tuple_to_upscaling(["foo", "bar", 2, True, 14])
      self.assertEqual(source["name"], "foo")
      self.assertEqual(source["source"], "bar")
      self.assertEqual(source["scale"], 2)
      self.assertEqual(source["half"], True)
      self.assertEqual(source["opset"], 14)
