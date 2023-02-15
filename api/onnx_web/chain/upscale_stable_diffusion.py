from logging import getLogger
from os import path

import torch
from diffusers import StableDiffusionUpscalePipeline
from PIL import Image

from ..diffusion.pipeline_onnx_stable_diffusion_upscale import (
    OnnxStableDiffusionUpscalePipeline,
)
from ..params import DeviceParams, ImageParams, StageParams, UpscaleParams
from ..server.device_pool import JobContext, ProgressCallback
from ..utils import ServerContext, run_gc

logger = getLogger(__name__)


def load_stable_diffusion(
    server: ServerContext, upscale: UpscaleParams, device: DeviceParams
):
    model_path = path.join(server.model_path, upscale.upscale_model)

    cache_key = (model_path, upscale.format)
    cache_pipe = server.cache.get("diffusion", cache_key)

    if cache_pipe is not None:
        logger.debug("reusing existing Stable Diffusion upscale pipeline")
        return cache_pipe

    if upscale.format == "onnx":
        logger.debug(
            "loading Stable Diffusion upscale ONNX model from %s, using provider %s",
            model_path,
            device.provider,
        )
        pipe = OnnxStableDiffusionUpscalePipeline.from_pretrained(
            model_path,
            provider=device.ort_provider(),
            sess_options=device.sess_options(),
        )
    else:
        logger.debug(
            "loading Stable Diffusion upscale model from %s, using provider %s",
            model_path,
            device.provider,
        )
        pipe = StableDiffusionUpscalePipeline.from_pretrained(
            model_path,
            provider=device.provider,
        )

    server.cache.set("diffusion", cache_key, pipe)
    run_gc()

    return pipe


def upscale_stable_diffusion(
    job: JobContext,
    server: ServerContext,
    _stage: StageParams,
    params: ImageParams,
    source: Image.Image,
    *,
    upscale: UpscaleParams,
    prompt: str = None,
    callback: ProgressCallback = None,
    **kwargs,
) -> Image.Image:
    prompt = prompt or params.prompt
    logger.info("upscaling with Stable Diffusion, %s steps: %s", params.steps, prompt)

    pipeline = load_stable_diffusion(server, upscale, job.get_device())
    generator = torch.manual_seed(params.seed)

    return pipeline(
        params.prompt,
        source,
        generator=generator,
        num_inference_steps=params.steps,
        callback=callback,
    ).images[0]
