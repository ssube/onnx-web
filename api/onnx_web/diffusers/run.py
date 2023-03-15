from logging import getLogger
from typing import Any, List

import numpy as np
import torch
from diffusers import OnnxStableDiffusionImg2ImgPipeline, OnnxStableDiffusionPipeline
from PIL import Image

from ..chain import blend_mask, upscale_outpaint
from ..chain.base import ChainProgress
from ..output import save_image, save_params
from ..params import Border, ImageParams, Size, StageParams, UpscaleParams
from ..server import ServerContext
from ..upscale import run_upscale_correction
from ..utils import run_gc
from ..worker import WorkerContext
from .load import get_latents_from_seed, load_pipeline
from .utils import get_loras_from_prompt

logger = getLogger(__name__)


def run_txt2img_pipeline(
    job: WorkerContext,
    server: ServerContext,
    params: ImageParams,
    size: Size,
    outputs: List[str],
    upscale: UpscaleParams,
) -> None:
    latents = get_latents_from_seed(params.seed, size, batch=params.batch)

    (prompt, loras) = get_loras_from_prompt(params.prompt)
    params.prompt = prompt

    pipe = load_pipeline(
        server,
        OnnxStableDiffusionPipeline,
        params.model,
        params.scheduler,
        job.get_device(),
        params.lpw,
        params.inversion,
        loras,
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
            num_images_per_prompt=params.batch,
            num_inference_steps=params.steps,
            eta=params.eta,
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
            num_images_per_prompt=params.batch,
            num_inference_steps=params.steps,
            eta=params.eta,
            callback=progress,
        )

    for image, output in zip(result.images, outputs):
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

    run_gc([job.get_device()])

    logger.info("finished txt2img job: %s", dest)


def run_img2img_pipeline(
    job: WorkerContext,
    server: ServerContext,
    params: ImageParams,
    outputs: List[str],
    upscale: UpscaleParams,
    source: Image.Image,
    strength: float,
) -> None:
    (prompt, loras) = get_loras_from_prompt(params.prompt)
    params.prompt = prompt

    pipe = load_pipeline(
        server,
        OnnxStableDiffusionImg2ImgPipeline,
        params.model,
        params.scheduler,
        job.get_device(),
        params.lpw,
        params.inversion,
        loras,
    )
    progress = job.get_progress_callback()
    if params.lpw:
        logger.debug("using LPW pipeline for img2img")
        rng = torch.manual_seed(params.seed)
        result = pipe.img2img(
            source,
            params.prompt,
            generator=rng,
            guidance_scale=params.cfg,
            negative_prompt=params.negative_prompt,
            num_images_per_prompt=params.batch,
            num_inference_steps=params.steps,
            strength=strength,
            eta=params.eta,
            callback=progress,
        )
    else:
        rng = np.random.RandomState(params.seed)
        result = pipe(
            params.prompt,
            source,
            generator=rng,
            guidance_scale=params.cfg,
            negative_prompt=params.negative_prompt,
            num_images_per_prompt=params.batch,
            num_inference_steps=params.steps,
            strength=strength,
            eta=params.eta,
            callback=progress,
        )

    for image, output in zip(result.images, outputs):
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
        size = Size(*source.size)
        save_params(server, output, params, size, upscale=upscale)

    run_gc([job.get_device()])

    logger.info("finished img2img job: %s", dest)


def run_inpaint_pipeline(
    job: WorkerContext,
    server: ServerContext,
    params: ImageParams,
    size: Size,
    outputs: List[str],
    upscale: UpscaleParams,
    source: Image.Image,
    mask: Image.Image,
    border: Border,
    noise_source: Any,
    mask_filter: Any,
    fill_color: str,
    tile_order: str,
) -> None:
    progress = job.get_progress_callback()
    stage = StageParams(tile_order=tile_order)

    # calling the upscale_outpaint stage directly needs accumulating progress
    progress = ChainProgress.from_progress(progress)

    logger.debug("applying mask filter and generating noise source")
    image = upscale_outpaint(
        job,
        server,
        stage,
        params,
        source,
        border=border,
        stage_mask=mask,
        fill_color=fill_color,
        mask_filter=mask_filter,
        noise_source=noise_source,
        callback=progress,
    )

    image = run_upscale_correction(
        job, server, stage, params, image, upscale=upscale, callback=progress
    )

    dest = save_image(server, outputs[0], image)
    save_params(server, outputs[0], params, size, upscale=upscale, border=border)

    del image

    run_gc([job.get_device()])

    logger.info("finished inpaint job: %s", dest)


def run_upscale_pipeline(
    job: WorkerContext,
    server: ServerContext,
    params: ImageParams,
    size: Size,
    outputs: List[str],
    upscale: UpscaleParams,
    source: Image.Image,
) -> None:
    progress = job.get_progress_callback()
    stage = StageParams()

    image = run_upscale_correction(
        job, server, stage, params, source, upscale=upscale, callback=progress
    )

    dest = save_image(server, outputs[0], image)
    save_params(server, outputs[0], params, size, upscale=upscale)

    del image

    run_gc([job.get_device()])

    logger.info("finished upscale job: %s", dest)


def run_blend_pipeline(
    job: WorkerContext,
    server: ServerContext,
    params: ImageParams,
    size: Size,
    outputs: List[str],
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
        stage_mask=mask,
        callback=progress,
    )
    image = image.convert("RGB")

    image = run_upscale_correction(
        job, server, stage, params, image, upscale=upscale, callback=progress
    )

    dest = save_image(server, outputs[0], image)
    save_params(server, outputs[0], params, size, upscale=upscale)

    del image

    run_gc([job.get_device()])

    logger.info("finished blend job: %s", dest)
