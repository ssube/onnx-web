from multiprocessing import Queue, Value
from os import path
from typing import List
from unittest import skipUnless

from onnx_web.params import DeviceParams
from onnx_web.worker.context import WorkerContext


def test_needs_models(models: List[str]):
    return skipUnless(
        all([path.exists(model) for model in models]), "model does not exist"
    )


def test_needs_onnx_models(models: List[str]):
    return skipUnless(
        all([path.exists(f"{model}.onnx") for model in models]), "model does not exist"
    )


def test_device() -> DeviceParams:
    return DeviceParams("cpu", "CPUExecutionProvider")


def test_worker() -> WorkerContext:
    cancel = Value("L", 0)
    logs = Queue()
    pending = Queue()
    progress = Queue()
    active = Value("L", 0)
    idle = Value("L", 0)

    return WorkerContext(
        "test",
        test_device(),
        cancel,
        logs,
        pending,
        progress,
        active,
        idle,
        3,
        0.1,
    )


TEST_MODEL_CORRECTION_CODEFORMER = "../models/.cache/correction-codeformer.pth"
TEST_MODEL_DIFFUSION_SD15 = "../models/stable-diffusion-onnx-v1-5"
TEST_MODEL_DIFFUSION_SD15_INPAINT = "../models/stable-diffusion-onnx-v1-inpainting"
TEST_MODEL_UPSCALING_SWINIR = "../models/.cache/upscaling-swinir.pth"
