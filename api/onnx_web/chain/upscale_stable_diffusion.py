from logging import getLogger
from os import path

import torch
from diffusers import StableDiffusionUpscalePipeline
from PIL import Image

from ..device_pool import JobContext
from ..diffusion.pipeline_onnx_stable_diffusion_upscale import (
    OnnxStableDiffusionUpscalePipeline,
)
from ..params import DeviceParams, ImageParams, StageParams, UpscaleParams
from ..utils import ServerContext, run_gc

logger = getLogger(__name__)


last_pipeline_instance = None
last_pipeline_params = (None, None)


def load_stable_diffusion(
    ctx: ServerContext, upscale: UpscaleParams, device: DeviceParams
):
    global last_pipeline_instance
    global last_pipeline_params

    model_path = path.join(ctx.model_path, upscale.upscale_model)
    cache_params = (model_path, upscale.format)

    if last_pipeline_instance is not None and cache_params == last_pipeline_params:
        logger.debug("reusing existing Stable Diffusion upscale pipeline")
        return last_pipeline_instance

    if upscale.format == "onnx":
        logger.debug(
            "loading Stable Diffusion upscale ONNX model from %s, using provider %s",
            model_path,
            device.provider,
        )
        pipeline = OnnxStableDiffusionUpscalePipeline.from_pretrained(
            model_path, provider=device.provider, provider_options=device.options
        )
    else:
        logger.debug(
            "loading Stable Diffusion upscale model from %s, using provider %s",
            model_path,
            device.provider,
        )
        pipeline = StableDiffusionUpscalePipeline.from_pretrained(
            model_path, provider=device.provider
        )

    last_pipeline_instance = pipeline
    last_pipeline_params = cache_params
    run_gc()

    return pipeline


def upscale_stable_diffusion(
    job: JobContext,
    server: ServerContext,
    _stage: StageParams,
    params: ImageParams,
    source: Image.Image,
    *,
    upscale: UpscaleParams,
    prompt: str = None,
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
    ).images[0]
