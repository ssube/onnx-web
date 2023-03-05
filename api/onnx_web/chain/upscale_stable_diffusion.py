from logging import getLogger
from os import path
from typing import Optional

import torch
from diffusers import StableDiffusionUpscalePipeline
from PIL import Image

from ..diffusers.load import optimize_pipeline
from ..diffusers.pipeline_onnx_stable_diffusion_upscale import (
    OnnxStableDiffusionUpscalePipeline,
)
from ..params import DeviceParams, ImageParams, StageParams, UpscaleParams
from ..server import ServerContext
from ..utils import run_gc
from ..worker import ProgressCallback, WorkerContext

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

    if not server.show_progress:
        pipe.set_progress_bar_config(disable=True)

    optimize_pipeline(server, pipe)

    server.cache.set("diffusion", cache_key, pipe)
    run_gc([device])

    return pipe


def upscale_stable_diffusion(
    job: WorkerContext,
    server: ServerContext,
    _stage: StageParams,
    params: ImageParams,
    source: Image.Image,
    *,
    upscale: UpscaleParams,
    stage_source: Optional[Image.Image] = None,
    callback: Optional[ProgressCallback] = None,
    **kwargs,
) -> Image.Image:
    params = params.with_args(**kwargs)
    upscale = upscale.with_args(**kwargs)
    source = stage_source or source
    logger.info(
        "upscaling with Stable Diffusion, %s steps: %s", params.steps, params.prompt
    )

    pipeline = load_stable_diffusion(server, upscale, job.get_device())
    generator = torch.manual_seed(params.seed)

    return pipeline(
        params.prompt,
        source,
        generator=generator,
        guidance_scale=params.cfg,
        negative_prompt=params.negative_prompt,
        num_inference_steps=params.steps,
        eta=params.eta,
        callback=callback,
    ).images[0]
