import unittest
from os import path
from unittest import mock
from unittest.mock import MagicMock, patch

from onnx_web.constants import ONNX_MODEL
from onnx_web.convert.utils import (
    DEFAULT_OPSET,
    ConversionContext,
    build_cache_paths,
    download_progress,
    fix_diffusion_name,
    get_first_exists,
    load_tensor,
    load_torch,
    remove_prefix,
    resolve_tensor,
    source_format,
    tuple_to_correction,
    tuple_to_diffusion,
    tuple_to_source,
    tuple_to_upscaling,
)
from tests.helpers import TEST_MODEL_UPSCALING_SWINIR, test_needs_models


class ConversionContextTests(unittest.TestCase):
    def test_from_environ(self):
        context = ConversionContext.from_environ()
        self.assertEqual(context.opset, DEFAULT_OPSET)

    def test_map_location(self):
        context = ConversionContext.from_environ()
        self.assertEqual(context.map_location.type, "cpu")


class DownloadProgressTests(unittest.TestCase):
    def test_download_example(self):
        path = download_progress("https://example.com", "/tmp/example-dot-com")
        self.assertEqual(path, "/tmp/example-dot-com")

    @patch("onnx_web.convert.utils.Path")
    @patch("onnx_web.convert.utils.requests")
    @patch("onnx_web.convert.utils.shutil")
    @patch("onnx_web.convert.utils.tqdm")
    def test_download_progress(self, mock_tqdm, mock_shutil, mock_requests, mock_path):
        source = "http://example.com/image.jpg"
        dest = "/path/to/destination/image.jpg"

        dest_path_mock = MagicMock()
        mock_path.return_value.expanduser.return_value.resolve.return_value = (
            dest_path_mock
        )
        dest_path_mock.exists.return_value = False
        dest_path_mock.absolute.return_value = "test"
        mock_requests.get.return_value.status_code = 200
        mock_requests.get.return_value.headers.get.return_value = "1000"
        mock_tqdm.wrapattr.return_value.__enter__.return_value = MagicMock()

        result = download_progress(source, dest)

        mock_path.assert_called_once_with(dest)
        dest_path_mock.parent.mkdir.assert_called_once_with(parents=True, exist_ok=True)
        dest_path_mock.open.assert_called_once_with("wb")
        mock_shutil.copyfileobj.assert_called_once()
        self.assertEqual(result, str(dest_path_mock.absolute.return_value))


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
        result = source_format(
            {
                "format": "foo",
            }
        )
        self.assertEqual(result, "foo")

    def test_source_known_extension(self):
        result = source_format(
            {
                "source": "foo.safetensors",
            }
        )
        self.assertEqual(result, "safetensors")

    def test_source_unknown_extension(self):
        result = source_format({"source": "foo.none"})
        self.assertEqual(result, None)

    def test_incomplete_model(self):
        self.assertIsNone(source_format({}))


class RemovePrefixTests(unittest.TestCase):
    def test_with_prefix(self):
        self.assertEqual(remove_prefix("foo.bar", "foo"), ".bar")

    def test_without_prefix(self):
        self.assertEqual(remove_prefix("foo.bar", "bin"), "foo.bar")


class ResolveTensorTests(unittest.TestCase):
    @test_needs_models([TEST_MODEL_UPSCALING_SWINIR])
    def test_resolve_existing(self):
        self.assertEqual(
            resolve_tensor("../models/.cache/upscaling-swinir"),
            TEST_MODEL_UPSCALING_SWINIR,
        )

    def test_resolve_missing(self):
        self.assertIsNone(resolve_tensor("missing"))


TORCH_MODEL = "model.pth"


class LoadTorchTests(unittest.TestCase):
    @patch("onnx_web.convert.utils.logger")
    @patch("onnx_web.convert.utils.torch")
    def test_load_torch_with_torch_load(self, mock_torch, mock_logger):
        map_location = "cpu"
        checkpoint = MagicMock()
        mock_torch.load.return_value = checkpoint

        result = load_torch(TORCH_MODEL, map_location)

        mock_logger.debug.assert_called_once_with(
            "loading tensor with Torch: %s", TORCH_MODEL
        )
        mock_torch.load.assert_called_once_with(TORCH_MODEL, map_location=map_location)
        self.assertEqual(result, checkpoint)

    @patch("onnx_web.convert.utils.logger")
    @patch("onnx_web.convert.utils.torch")
    def test_load_torch_with_torch_jit_load(self, mock_torch, mock_logger):
        checkpoint = MagicMock()
        mock_torch.load.side_effect = Exception()
        mock_torch.jit.load.return_value = checkpoint

        result = load_torch(TORCH_MODEL)

        mock_logger.debug.assert_called_once_with(
            "loading tensor with Torch: %s", TORCH_MODEL
        )
        mock_logger.exception.assert_called_once_with(
            "error loading with Torch, trying with Torch JIT: %s", TORCH_MODEL
        )
        mock_torch.jit.load.assert_called_once_with(TORCH_MODEL)
        self.assertEqual(result, checkpoint)


class LoadTensorTests(unittest.TestCase):
    @patch("onnx_web.convert.utils.logger")
    @patch("onnx_web.convert.utils.path")
    @patch("onnx_web.convert.utils.torch")
    def test_load_tensor_with_no_extension(self, mock_torch, mock_path, mock_logger):
        name = "model"
        map_location = "cpu"
        checkpoint = MagicMock()
        mock_path.exists.return_value = True
        mock_path.splitext.side_effect = [("model", ""), ("model", ".safetensors")]
        mock_torch.load.return_value = checkpoint

        result = load_tensor(name, map_location)

        mock_logger.debug.assert_has_calls([mock.call("loading tensor: %s", name)])
        mock_path.splitext.assert_called_once_with(name)
        mock_path.exists.assert_called_once_with(name)
        mock_torch.load.assert_called_once_with(name, map_location=map_location)
        self.assertEqual(result, checkpoint)

    @patch("onnx_web.convert.utils.logger")
    @patch("onnx_web.convert.utils.environ")
    @patch("onnx_web.convert.utils.safetensors")
    def test_load_tensor_with_safetensors_extension(
        self, mock_safetensors, mock_environ, mock_logger
    ):
        name = "model.safetensors"
        checkpoint = MagicMock()
        mock_environ.__getitem__.return_value = "1"
        mock_safetensors.torch.load_file.return_value = checkpoint

        result = load_tensor(name)

        mock_logger.debug.assert_has_calls([mock.call("loading tensor: %s", name)])
        mock_safetensors.torch.load_file.assert_called_once_with(name, device="cpu")
        self.assertEqual(result, checkpoint)

    @patch("onnx_web.convert.utils.logger")
    @patch("onnx_web.convert.utils.torch")
    def test_load_tensor_with_pickle_extension(self, mock_torch, mock_logger):
        name = "model.pt"
        map_location = "cpu"
        checkpoint = MagicMock()
        mock_torch.load.side_effect = [checkpoint]

        result = load_tensor(name, map_location)

        mock_logger.debug.assert_has_calls([mock.call("loading tensor: %s", name)])
        mock_torch.load.assert_has_calls(
            [
                mock.call(name, map_location=map_location),
            ]
        )
        self.assertEqual(result, checkpoint)

    @patch("onnx_web.convert.utils.logger")
    @patch("onnx_web.convert.utils.torch")
    def test_load_tensor_with_onnx_extension(self, mock_torch, mock_logger):
        map_location = "cpu"
        checkpoint = MagicMock()
        mock_torch.load.side_effect = [checkpoint]

        result = load_tensor(ONNX_MODEL, map_location)

        mock_logger.debug.assert_has_calls(
            [mock.call("loading tensor: %s", ONNX_MODEL)]
        )
        mock_logger.warning.assert_called_once_with(
            "tensor has ONNX extension, attempting to use PyTorch anyways: %s", "onnx"
        )
        mock_torch.load.assert_has_calls(
            [
                mock.call(ONNX_MODEL, map_location=map_location),
            ]
        )
        self.assertEqual(result, checkpoint)

    @patch("onnx_web.convert.utils.logger")
    @patch("onnx_web.convert.utils.torch")
    def test_load_tensor_with_unknown_extension(self, mock_torch, mock_logger):
        name = "model.xyz"
        map_location = "cpu"
        checkpoint = MagicMock()
        mock_torch.load.side_effect = [checkpoint]

        result = load_tensor(name, map_location)

        mock_logger.debug.assert_has_calls([mock.call("loading tensor: %s", name)])
        mock_logger.warning.assert_called_once_with(
            "unknown tensor type, falling back to PyTorch: %s", "xyz"
        )
        mock_torch.load.assert_has_calls(
            [
                mock.call(name, map_location=map_location),
            ]
        )
        self.assertEqual(result, checkpoint)

    @patch("onnx_web.convert.utils.logger")
    @patch("onnx_web.convert.utils.torch")
    def test_load_tensor_with_error_loading_tensor(self, mock_torch, mock_logger):
        name = "model"
        map_location = "cpu"
        mock_torch.load.side_effect = Exception()

        with self.assertRaises(ValueError):
            load_tensor(name, map_location)


class FixDiffusionNameTests(unittest.TestCase):
    def test_fix_diffusion_name_with_valid_name(self):
        name = "diffusion-model"
        result = fix_diffusion_name(name)
        self.assertEqual(result, name)

    @patch("onnx_web.convert.utils.logger")
    def test_fix_diffusion_name_with_invalid_name(self, logger):
        name = "model"
        expected_result = "diffusion-model"
        result = fix_diffusion_name(name)

        self.assertEqual(result, expected_result)
        logger.warning.assert_called_once_with(
            "diffusion models must have names starting with diffusion- to be recognized by the server: %s does not match",
            name,
        )


class BuildCachePathsTests(unittest.TestCase):
    def test_build_cache_paths_without_format(self):
        client = "client1"
        cache = "/path/to/cache"

        conversion = ConversionContext(cache_path=cache)
        result = build_cache_paths(conversion, ONNX_MODEL, client, cache)

        expected_paths = [
            path.join("/path/to/cache", ONNX_MODEL),
            path.join("/path/to/cache/client1", ONNX_MODEL),
        ]
        self.assertEqual(result, expected_paths)

    def test_build_cache_paths_with_format(self):
        name = "model"
        client = "client2"
        cache = "/path/to/cache"
        model_format = "onnx"

        conversion = ConversionContext(cache_path=cache)
        result = build_cache_paths(conversion, name, client, cache, model_format)

        expected_paths = [
            path.join("/path/to/cache", ONNX_MODEL),
            path.join("/path/to/cache/client2", ONNX_MODEL),
        ]
        self.assertEqual(result, expected_paths)

    def test_build_cache_paths_with_existing_extension(self):
        client = "client3"
        cache = "/path/to/cache"
        model_format = "onnx"

        conversion = ConversionContext(cache_path=cache)
        result = build_cache_paths(conversion, TORCH_MODEL, client, cache, model_format)

        expected_paths = [
            path.join("/path/to/cache", TORCH_MODEL),
            path.join("/path/to/cache/client3", TORCH_MODEL),
        ]
        self.assertEqual(result, expected_paths)

    def test_build_cache_paths_with_empty_extension(self):
        name = "model"
        client = "client4"
        cache = "/path/to/cache"
        model_format = "onnx"

        conversion = ConversionContext(cache_path=cache)
        result = build_cache_paths(conversion, name, client, cache, model_format)

        expected_paths = [
            path.join("/path/to/cache", ONNX_MODEL),
            path.join("/path/to/cache/client4", ONNX_MODEL),
        ]
        self.assertEqual(result, expected_paths)


class GetFirstExistsTests(unittest.TestCase):
    @patch("onnx_web.convert.utils.path")
    @patch("onnx_web.convert.utils.logger")
    def test_get_first_exists_with_existing_path(self, mock_logger, mock_path):
        paths = ["path1", "path2", "path3"]
        mock_path.exists.side_effect = [False, True, False]
        mock_path.return_value = MagicMock()

        result = get_first_exists(paths)

        mock_logger.debug.assert_called_once_with(
            "model already exists in cache, skipping fetch: %s", "path2"
        )
        self.assertEqual(result, "path2")

    @patch("onnx_web.convert.utils.path")
    @patch("onnx_web.convert.utils.logger")
    def test_get_first_exists_with_no_existing_path(self, mock_logger, mock_path):
        paths = ["path1", "path2", "path3"]
        mock_path.exists.return_value = False

        result = get_first_exists(paths)

        mock_logger.debug.assert_not_called()
        self.assertIsNone(result)
