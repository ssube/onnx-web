import unittest

from onnx_web.convert.utils import (
    DEFAULT_OPSET,
    ConversionContext,
    download_progress,
    remove_prefix,
    resolve_tensor,
    source_format,
    tuple_to_correction,
    tuple_to_diffusion,
    tuple_to_source,
    tuple_to_upscaling,
)
from tests.helpers import (
    TEST_MODEL_DIFFUSION_SD15,
    TEST_MODEL_UPSCALING_SWINIR,
    test_needs_models,
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


class SourceFormatTests(unittest.TestCase):
   def test_with_format(self):
      result = source_format({
         "format": "foo",
      })
      self.assertEqual(result, "foo")

   def test_source_known_extension(self):
      result = source_format({
         "source": "foo.safetensors",
      })
      self.assertEqual(result, "safetensors")

   def test_source_unknown_extension(self):
      result = source_format({
         "source": "foo.none"
      })
      self.assertEqual(result, None)

   def test_incomplete_model(self):
      self.assertIsNone(source_format({}))


class RemovePrefixTests(unittest.TestCase):
   def test_with_prefix(self):
      self.assertEqual(remove_prefix("foo.bar", "foo"), ".bar")

   def test_without_prefix(self):
      self.assertEqual(remove_prefix("foo.bar", "bin"), "foo.bar")


class LoadTorchTests(unittest.TestCase):
   pass


class LoadTensorTests(unittest.TestCase):
   pass


class ResolveTensorTests(unittest.TestCase):
   @test_needs_models([TEST_MODEL_UPSCALING_SWINIR])
   def test_resolve_existing(self):
      self.assertEqual(resolve_tensor("../models/.cache/upscaling-swinir"), TEST_MODEL_UPSCALING_SWINIR)

   def test_resolve_missing(self):
      self.assertIsNone(resolve_tensor("missing"))
