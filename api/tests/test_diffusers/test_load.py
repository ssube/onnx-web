import unittest

from onnx_web.diffusers.load import get_available_pipelines, get_pipeline_schedulers, get_scheduler_name
from diffusers import DDIMScheduler


class TestAvailablePipelines(unittest.TestCase):
  def test_available_pipelines(self):
    pipelines = get_available_pipelines()

    self.assertIn("txt2img", pipelines)


class TestPipelineSchedulers(unittest.TestCase):
  def test_pipeline_schedulers(self):
    schedulers = get_pipeline_schedulers()

    self.assertIn("euler-a", schedulers)


class TestSchedulerNames(unittest.TestCase):
  def test_valid_name(self):
    scheduler = get_scheduler_name(DDIMScheduler)

    self.assertEqual("ddim", scheduler)

  def test_missing_names(self):
    self.assertIsNone(get_scheduler_name("test"))


class TestOptimizePipeline(unittest.TestCase):
  def test_auto_attention_slicing(self):
    pass

  def test_max_attention_slicing(self):
    pass

  def test_vae_slicing(self):
    pass

  def test_cpu_offload_sequential(self):
    pass

  def test_cpu_offload_model(self):
    pass

  def test_memory_efficient_attention(self):
    pass


class TestPatchPipeline(unittest.TestCase):
  def test_expand_not_lpw(self):
    pass

  def test_unet_wrapper_not_xl(self):
    pass

  def test_vae_wrapper(self):
    pass