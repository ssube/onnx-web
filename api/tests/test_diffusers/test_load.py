import unittest
from unittest.mock import MagicMock
from os import path

import torch
from diffusers import DDIMScheduler, OnnxRuntimeModel

from onnx_web.diffusers.load import (
    get_available_pipelines,
    get_pipeline_schedulers,
    get_scheduler_name,
    load_controlnet,
    load_text_encoders,
    load_unet,
    load_vae,
    optimize_pipeline,
    patch_pipeline,
)
from onnx_web.diffusers.patches.unet import UNetWrapper
from onnx_web.diffusers.patches.vae import VAEWrapper
from onnx_web.models.meta import NetworkModel
from onnx_web.params import DeviceParams, ImageParams
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
        session = MagicMock()
        session.get_inputs.return_value = []

        server = ServerContext()
        pipeline = MockPipeline()
        pipeline.unet = OnnxRuntimeModel(model=session)

        patch_pipeline(
            server,
            pipeline,
            None,
            ImageParams("test", "txt2img", "ddim", "test", 1.0, 10, 1),
        )
        self.assertTrue(isinstance(pipeline.unet, UNetWrapper))

    def test_unet_wrapper_xl(self):
        session = MagicMock()
        session.get_inputs.return_value = []

        server = ServerContext()
        pipeline = MockPipeline()
        pipeline.unet = OnnxRuntimeModel(model=session)

        patch_pipeline(
            server,
            pipeline,
            None,
            ImageParams("test", "txt2img-sdxl", "ddim", "test", 1.0, 10, 1),
        )
        self.assertTrue(isinstance(pipeline.unet, UNetWrapper))

    def test_vae_wrapper(self):
        session = MagicMock()
        session.get_inputs.return_value = []

        server = ServerContext()
        pipeline = MockPipeline()
        pipeline.unet = OnnxRuntimeModel(model=session)

        patch_pipeline(
            server,
            pipeline,
            None,
            ImageParams("test", "txt2img", "ddim", "test", 1.0, 10, 1),
        )
        self.assertTrue(isinstance(pipeline.vae_decoder, VAEWrapper))
        self.assertTrue(isinstance(pipeline.vae_encoder, VAEWrapper))


class TestLoadControlNet(unittest.TestCase):
    @unittest.skipUnless(
        path.exists("../models/control/canny.onnx"), "model does not exist"
    )
    def test_load_existing(self):
        """
        Should load a model
        """
        components = load_controlnet(
            ServerContext(model_path="../models"),
            DeviceParams("cpu", "CPUExecutionProvider"),
            ImageParams(
                "test",
                "txt2img",
                "ddim",
                "test",
                1.0,
                10,
                1,
                control=NetworkModel("canny", "control"),
            ),
        )
        self.assertIn("controlnet", components)

    def test_load_missing(self):
        """
        Should throw
        """
        components = {}
        try:
            components = load_controlnet(
                ServerContext(),
                DeviceParams("cpu", "CPUExecutionProvider"),
                ImageParams(
                    "test",
                    "txt2img",
                    "ddim",
                    "test",
                    1.0,
                    10,
                    1,
                    control=NetworkModel("missing", "control"),
                ),
            )
        except Exception:
            self.assertNotIn("controlnet", components)
            return

        self.fail()


class TestLoadTextEncoders(unittest.TestCase):
    @unittest.skipUnless(
        path.exists("../models/stable-diffusion-onnx-v1-5/text_encoder/model.onnx"),
        "model does not exist",
    )
    def test_load_embeddings(self):
        """
        Should add the token to tokenizer
        Should increase the encoder dims
        """
        components = load_text_encoders(
            ServerContext(model_path="../models"),
            DeviceParams("cpu", "CPUExecutionProvider"),
            "../models/stable-diffusion-onnx-v1-5",
            [
                # TODO: add some embeddings
            ],
            [],
            torch.float32,
            ImageParams("test", "txt2img", "ddim", "test", 1.0, 10, 1),
        )
        self.assertIn("text_encoder", components)

    def test_load_embeddings_xl(self):
        pass

    @unittest.skipUnless(
        path.exists("../models/stable-diffusion-onnx-v1-5/text_encoder/model.onnx"),
        "model does not exist",
    )
    def test_load_loras(self):
        components = load_text_encoders(
            ServerContext(model_path="../models"),
            DeviceParams("cpu", "CPUExecutionProvider"),
            "../models/stable-diffusion-onnx-v1-5",
            [],
            [
                # TODO: add some loras
            ],
            torch.float32,
            ImageParams("test", "txt2img", "ddim", "test", 1.0, 10, 1),
        )
        self.assertIn("text_encoder", components)

    def test_load_loras_xl(self):
        pass


class TestLoadUnet(unittest.TestCase):
    @unittest.skipUnless(
        path.exists("../models/stable-diffusion-onnx-v1-5/unet/model.onnx"),
        "model does not exist",
    )
    def test_load_unet_loras(self):
        components = load_unet(
            ServerContext(model_path="../models"),
            DeviceParams("cpu", "CPUExecutionProvider"),
            "../models/stable-diffusion-onnx-v1-5",
            [
                # TODO: add some loras
            ],
            "unet",
            ImageParams("test", "txt2img", "ddim", "test", 1.0, 10, 1),
        )
        self.assertIn("unet", components)

    def test_load_unet_loras_xl(self):
        pass

    @unittest.skipUnless(
        path.exists("../models/stable-diffusion-onnx-v1-5/cnet/model.onnx"),
        "model does not exist",
    )
    def test_load_cnet_loras(self):
        components = load_unet(
            ServerContext(model_path="../models"),
            DeviceParams("cpu", "CPUExecutionProvider"),
            "../models/stable-diffusion-onnx-v1-5",
            [
                # TODO: add some loras
            ],
            "cnet",
            ImageParams("test", "txt2img", "ddim", "test", 1.0, 10, 1),
        )
        self.assertIn("unet", components)


class TestLoadVae(unittest.TestCase):
    @unittest.skipUnless(
        path.exists("../models/upscaling-stable-diffusion-x4/vae/model.onnx"),
        "model does not exist",
    )
    def test_load_single(self):
        """
        Should return single component
        """
        components = load_vae(
            ServerContext(model_path="../models"),
            DeviceParams("cpu", "CPUExecutionProvider"),
            "../models/upscaling-stable-diffusion-x4",
            ImageParams("test", "txt2img", "ddim", "test", 1.0, 10, 1),
        )
        self.assertIn("vae", components)
        self.assertNotIn("vae_decoder", components)
        self.assertNotIn("vae_encoder", components)

    @unittest.skipUnless(
        path.exists("../models/stable-diffusion-onnx-v1-5/vae_encoder/model.onnx"),
        "model does not exist",
    )
    def test_load_split(self):
        """
        Should return split encoder/decoder
        """
        components = load_vae(
            ServerContext(model_path="../models"),
            DeviceParams("cpu", "CPUExecutionProvider"),
            "../models/stable-diffusion-onnx-v1-5",
            ImageParams("test", "txt2img", "ddim", "test", 1.0, 10, 1),
        )
        self.assertNotIn("vae", components)
        self.assertIn("vae_decoder", components)
        self.assertIn("vae_encoder", components)
