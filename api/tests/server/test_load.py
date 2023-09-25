import unittest

from onnx_web.server.load import (
    get_available_platforms,
    get_config_params,
    get_correction_models,
    get_diffusion_models,
    get_extra_hashes,
    get_extra_strings,
    get_highres_methods,
    get_mask_filters,
    get_network_models,
    get_noise_sources,
    get_source_filters,
    get_upscaling_models,
    get_wildcard_data,
)


class ConfigParamTests(unittest.TestCase):
    def test_before_setup(self):
        params = get_config_params()
        self.assertIsNotNone(params)

class AvailablePlatformTests(unittest.TestCase):
    def test_before_setup(self):
        platforms = get_available_platforms()
        self.assertIsNotNone(platforms)

class CorrectModelTests(unittest.TestCase):
    def test_before_setup(self):
        models = get_correction_models()
        self.assertIsNotNone(models)

class DiffusionModelTests(unittest.TestCase):
    def test_before_setup(self):
        models = get_diffusion_models()
        self.assertIsNotNone(models)

class NetworkModelTests(unittest.TestCase):
    def test_before_setup(self):
        models = get_network_models()
        self.assertIsNotNone(models)

class UpscalingModelTests(unittest.TestCase):
    def test_before_setup(self):
        models = get_upscaling_models()
        self.assertIsNotNone(models)

class WildcardDataTests(unittest.TestCase):
    def test_before_setup(self):
        wildcards = get_wildcard_data()
        self.assertIsNotNone(wildcards)

class ExtraStringsTests(unittest.TestCase):
    def test_before_setup(self):
        strings = get_extra_strings()
        self.assertIsNotNone(strings)

class ExtraHashesTests(unittest.TestCase):
    def test_before_setup(self):
        hashes = get_extra_hashes()
        self.assertIsNotNone(hashes)

class HighresMethodTests(unittest.TestCase):
    def test_before_setup(self):
        methods = get_highres_methods()
        self.assertIsNotNone(methods)

class MaskFilterTests(unittest.TestCase):
    def test_before_setup(self):
        filters = get_mask_filters()
        self.assertIsNotNone(filters)

class NoiseSourceTests(unittest.TestCase):
    def test_before_setup(self):
        sources = get_noise_sources()
        self.assertIsNotNone(sources)

class SourceFilterTests(unittest.TestCase):
    def test_before_setup(self):
        filters = get_source_filters()
        self.assertIsNotNone(filters)
