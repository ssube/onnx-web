from logging import getLogger
from typing import Any, List

import numpy as np
import torch
from diffusers import OnnxStableDiffusionImg2ImgPipeline, OnnxStableDiffusionPipeline
from PIL import Image, ImageChops

from onnx_web.chain import blend_mask
from onnx_web.chain.base import ChainProgress

from ..chain import upscale_outpaint
from ..output import save_image, save_params
from ..params import Border, ImageParams, Size, StageParams
from ..server.device_pool import JobContext
from ..server.upscale import UpscaleParams, run_upscale_correction
from ..utils import ServerContext, run_gc
from .load import get_latents_from_seed, load_pipeline

logger = getLogger(__name__)


def run_txt2img_pipeline(
    job: JobContext,
    server: ServerContext,
    params: ImageParams,
    size: Size,
    output: str,
    upscale: UpscaleParams,
) -> None:
    latents = get_latents_from_seed(params.seed, size)
    pipe = load_pipeline(
        server,
        OnnxStableDiffusionPipeline,
        params.model,
        params.scheduler,
        job.get_device(),
        params.lpw,
    )
    progress = job.get_progress_callback()

    if params.lpw:
        logger.debug("using LPW pipeline for txt2img")
        rng = torch.manual_seed(params.seed)
        result = pipe.text2img(
            params.prompt,
            height=size.height,
            width=size.width,
            generator=rng,
            guidance_scale=params.cfg,
            latents=latents,
            negative_prompt=params.negative_prompt,
            num_inference_steps=params.steps,
            callback=progress,
        )
    else:
        rng = np.random.RandomState(params.seed)
        result = pipe(
            params.prompt,
            height=size.height,
            width=size.width,
            generator=rng,
            guidance_scale=params.cfg,
            latents=latents,
            negative_prompt=params.negative_prompt,
            num_inference_steps=params.steps,
            callback=progress,
        )

    image = result.images[0]
    image = run_upscale_correction(
        job,
        server,
        StageParams(),
        params,
        image,
        upscale=upscale,
        callback=progress,
    )

    dest = save_image(server, output, image)
    save_params(server, output, params, size, upscale=upscale)

    del image
    del result
    run_gc()

    logger.info("finished txt2img job: %s", dest)


def run_img2img_pipeline(
    job: JobContext,
    server: ServerContext,
    params: ImageParams,
    output: str,
    upscale: UpscaleParams,
    source_image: Image.Image,
    strength: float,
) -> None:
    pipe = load_pipeline(
        server,
        OnnxStableDiffusionImg2ImgPipeline,
        params.model,
        params.scheduler,
        job.get_device(),
        params.lpw,
    )
    progress = job.get_progress_callback()
    if params.lpw:
        logger.debug("using LPW pipeline for img2img")
        rng = torch.manual_seed(params.seed)
        result = pipe.img2img(
            source_image,
            params.prompt,
            generator=rng,
            guidance_scale=params.cfg,
            negative_prompt=params.negative_prompt,
            num_inference_steps=params.steps,
            strength=strength,
            callback=progress,
        )
    else:
        rng = np.random.RandomState(params.seed)
        result = pipe(
            params.prompt,
            source_image,
            generator=rng,
            guidance_scale=params.cfg,
            negative_prompt=params.negative_prompt,
            num_inference_steps=params.steps,
            strength=strength,
            callback=progress,
        )

    image = result.images[0]
    image = run_upscale_correction(
        job,
        server,
        StageParams(),
        params,
        image,
        upscale=upscale,
        callback=progress,
    )

    dest = save_image(server, output, image)
    size = Size(*source_image.size)
    save_params(server, output, params, size, upscale=upscale)

    del image
    del result
    run_gc()

    logger.info("finished img2img job: %s", dest)


def run_inpaint_pipeline(
    job: JobContext,
    server: ServerContext,
    params: ImageParams,
    size: Size,
    output: str,
    upscale: UpscaleParams,
    source_image: Image.Image,
    mask_image: Image.Image,
    border: Border,
    noise_source: Any,
    mask_filter: Any,
    strength: float,
    fill_color: str,
    tile_order: str,
) -> None:
    # device = job.get_device()
    progress = job.get_progress_callback()
    stage = StageParams(tile_order=tile_order)

    # calling the upscale_outpaint stage directly needs accumulating progress
    progress = ChainProgress.from_progress(progress)

    image = upscale_outpaint(
        job,
        server,
        stage,
        params,
        source_image,
        border=border,
        mask_image=mask_image,
        fill_color=fill_color,
        mask_filter=mask_filter,
        noise_source=noise_source,
        callback=progress,
    )
    logger.info("applying mask filter and generating noise source")

    if image.size == source_image.size:
        image = ImageChops.blend(source_image, image, strength)
    else:
        logger.info("output image size does not match source, skipping post-blend")

    image = run_upscale_correction(
        job, server, stage, params, image, upscale=upscale, callback=progress
    )

    dest = save_image(server, output, image)
    save_params(server, output, params, size, upscale=upscale, border=border)

    del image
    run_gc()

    logger.info("finished inpaint job: %s", dest)


def run_upscale_pipeline(
    job: JobContext,
    server: ServerContext,
    params: ImageParams,
    size: Size,
    output: str,
    upscale: UpscaleParams,
    source_image: Image.Image,
) -> None:
    # device = job.get_device()
    progress = job.get_progress_callback()
    stage = StageParams()

    image = run_upscale_correction(
        job, server, stage, params, source_image, upscale=upscale, callback=progress
    )

    dest = save_image(server, output, image)
    save_params(server, output, params, size, upscale=upscale)

    del image
    run_gc()

    logger.info("finished upscale job: %s", dest)


def run_blend_pipeline(
    job: JobContext,
    server: ServerContext,
    params: ImageParams,
    size: Size,
    output: str,
    upscale: UpscaleParams,
    sources: List[Image.Image],
    mask: Image.Image,
) -> None:
    progress = job.get_progress_callback()
    stage = StageParams()

    image = blend_mask(
        job,
        server,
        stage,
        params,
        sources=sources,
        mask=mask,
        callback=progress,
    )

    image = run_upscale_correction(
        job, server, stage, params, image, upscale=upscale, callback=progress
    )

    dest = save_image(server, output, image)
    save_params(server, output, params, size, upscale=upscale)

    del image
    run_gc()

    logger.info("finished blend job: %s", dest)
