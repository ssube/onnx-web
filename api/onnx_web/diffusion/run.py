from diffusers import (
    OnnxStableDiffusionPipeline,
    OnnxStableDiffusionImg2ImgPipeline,
)
from logging import getLogger
from PIL import Image, ImageChops
from typing import Any

from ..chain import (
    upscale_outpaint,
)
from ..device_pool import (
    JobContext,
)
from ..params import (
    ImageParams,
    Border,
    Size,
    StageParams,
)
from ..output import (
    save_image,
    save_params,
)
from ..upscale import (
    run_upscale_correction,
    UpscaleParams,
)
from ..utils import (
    run_gc,
    ServerContext,
)
from .load import (
    get_latents_from_seed,
    load_pipeline,
)

import numpy as np

logger = getLogger(__name__)


def run_txt2img_pipeline(
    job: JobContext,
    server: ServerContext,
    params: ImageParams,
    size: Size,
    output: str,
    upscale: UpscaleParams
) -> None:
    device = job.get_device()
    pipe = load_pipeline(OnnxStableDiffusionPipeline,
                         params.model, device.provider, params.scheduler, device=device.torch_device())

    latents = get_latents_from_seed(params.seed, size)
    rng = np.random.RandomState(params.seed)

    progress = job.get_progress_callback()
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
        job, server, StageParams(), params, image, upscale=upscale)

    dest = save_image(server, output, image)
    save_params(server, output, params, size, upscale=upscale)

    del image
    del result
    run_gc()

    logger.info('finished txt2img job: %s', dest)


def run_img2img_pipeline(
    job: JobContext,
    server: ServerContext,
    params: ImageParams,
    output: str,
    upscale: UpscaleParams,
    source_image: Image.Image,
    strength: float,
) -> None:
    device = job.get_device()
    pipe = load_pipeline(OnnxStableDiffusionImg2ImgPipeline,
                         params.model, device.provider, params.scheduler, device=device.torch_device())

    rng = np.random.RandomState(params.seed)

    progress = job.get_progress_callback()
    result = pipe(
        params.prompt,
        generator=rng,
        guidance_scale=params.cfg,
        image=source_image,
        negative_prompt=params.negative_prompt,
        num_inference_steps=params.steps,
        strength=strength,
        callback=progress,
    )
    image = result.images[0]
    image = run_upscale_correction(
        job, server, StageParams(), params, image, upscale=upscale)

    dest = save_image(server, output, image)
    size = Size(*source_image.size)
    save_params(server, output, params, size, upscale=upscale)

    del image
    del result
    run_gc()

    logger.info('finished img2img job: %s', dest)


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
) -> None:
    # device = job.get_device()
    # progress = job.get_progress_callback()
    stage = StageParams()

    image = upscale_outpaint(
        server,
        stage,
        params,
        source_image,
        border=border,
        mask_image=mask_image,
        fill_color=fill_color,
        mask_filter=mask_filter,
        noise_source=noise_source,
    )
    logger.info('applying mask filter and generating noise source')

    if image.size == source_image.size:
        image = ImageChops.blend(source_image, image, strength)
    else:
        logger.info(
            'output image size does not match source, skipping post-blend')

    image = run_upscale_correction(
        job, server, stage, params, image, upscale=upscale)

    dest = save_image(server, output, image)
    save_params(server, output, params, size, upscale=upscale, border=border)

    del image
    run_gc()

    logger.info('finished inpaint job: %s', dest)


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
    # progress = job.get_progress_callback()
    stage = StageParams()

    image = run_upscale_correction(
        job, server, stage, params, source_image, upscale=upscale)

    dest = save_image(server, output, image)
    save_params(server, output, params, size, upscale=upscale)

    del image
    run_gc()

    logger.info('finished upscale job: %s', dest)
