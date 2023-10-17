from os import path
from typing import List
from unittest import skipUnless

from onnx_web.params import DeviceParams


def test_needs_models(models: List[str]):
  return skipUnless(all([path.exists(model) for model in models]), "model does not exist")


def test_device() -> DeviceParams:
  return DeviceParams("cpu", "CPUExecutionProvider")


TEST_MODEL_DIFFUSION_SD15 = "../models/stable-diffusion-onnx-v1-5"
TEST_MODEL_UPSCALING_SWINIR = "../models/.cache/upscaling-swinir.pth"
