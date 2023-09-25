import unittest

from diffusers import DDIMScheduler

from onnx_web.diffusers.load import (
    get_available_pipelines,
    get_pipeline_schedulers,
    get_scheduler_name,
    optimize_pipeline,
    patch_pipeline,
)
from onnx_web.diffusers.patches.unet import UNetWrapper
from onnx_web.diffusers.patches.vae import VAEWrapper
from onnx_web.diffusers.utils import expand_prompt
from onnx_web.params import ImageParams
from onnx_web.server.context import ServerContext
from tests.mocks import MockPipeline


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
    server = ServerContext(
      optimizations=[
        "diffusers-attention-slicing-auto",
      ],
    )
    pipeline = MockPipeline()
    optimize_pipeline(server, pipeline)
    self.assertEqual(pipeline.slice_size, "auto")

  def test_max_attention_slicing(self):
    server = ServerContext(
      optimizations=[
        "diffusers-attention-slicing-max",
      ]
    )
    pipeline = MockPipeline()
    optimize_pipeline(server, pipeline)
    self.assertEqual(pipeline.slice_size, "max")

  def test_vae_slicing(self):
    server = ServerContext(
      optimizations=[
        "diffusers-vae-slicing",
      ]
    )
    pipeline = MockPipeline()
    optimize_pipeline(server, pipeline)
    self.assertEqual(pipeline.vae_slicing, True)

  def test_cpu_offload_sequential(self):
    server = ServerContext(
      optimizations=[
        "diffusers-cpu-offload-sequential",
      ]
    )
    pipeline = MockPipeline()
    optimize_pipeline(server, pipeline)
    self.assertEqual(pipeline.sequential_offload, True)

  def test_cpu_offload_model(self):
    server = ServerContext(
      optimizations=[
        "diffusers-cpu-offload-model",
      ]
    )
    pipeline = MockPipeline()
    optimize_pipeline(server, pipeline)
    self.assertEqual(pipeline.model_offload, True)

  def test_memory_efficient_attention(self):
    server = ServerContext(
      optimizations=[
        "diffusers-memory-efficient-attention",
      ]
    )
    pipeline = MockPipeline()
    optimize_pipeline(server, pipeline)
    self.assertEqual(pipeline.xformers, True)


class TestPatchPipeline(unittest.TestCase):
  def test_expand_not_lpw(self):
    """
    server = ServerContext()
    pipeline = MockPipeline()
    patch_pipeline(server, pipeline, None, ImageParams("test", "txt2img", "ddim", "test", 1.0, 10, 1))
    self.assertEqual(pipeline._encode_prompt, expand_prompt)
    """
    pass

  def test_unet_wrapper_not_xl(self):
    server = ServerContext()
    pipeline = MockPipeline()
    patch_pipeline(server, pipeline, None, ImageParams("test", "txt2img", "ddim", "test", 1.0, 10, 1))
    self.assertTrue(isinstance(pipeline.unet, UNetWrapper))

  def test_unet_wrapper_xl(self):
    server = ServerContext()
    pipeline = MockPipeline()
    patch_pipeline(server, pipeline, None, ImageParams("test", "txt2img-sdxl", "ddim", "test", 1.0, 10, 1))
    self.assertTrue(isinstance(pipeline.unet, UNetWrapper))

  def test_vae_wrapper(self):
    server = ServerContext()
    pipeline = MockPipeline()
    patch_pipeline(server, pipeline, None, ImageParams("test", "txt2img", "ddim", "test", 1.0, 10, 1))
    self.assertTrue(isinstance(pipeline.vae_decoder, VAEWrapper))
    self.assertTrue(isinstance(pipeline.vae_encoder, VAEWrapper))
