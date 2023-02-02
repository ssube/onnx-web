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
    ctx: ServerContext,
    params: ImageParams,
    size: Size,
    output: str,
    upscale: UpscaleParams
) -> None:
    pipe = load_pipeline(OnnxStableDiffusionPipeline,
                         params.model, params.provider, params.scheduler)

    latents = get_latents_from_seed(params.seed, size)
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
    )
    image = result.images[0]
    image = run_upscale_correction(
        ctx, StageParams(), params, image, upscale=upscale)

    dest = save_image(ctx, output, image)
    save_params(ctx, output, params, size, upscale=upscale)

    del image
    del result
    run_gc()

    logger.info('finished txt2img job: %s', dest)


def run_img2img_pipeline(
    ctx: ServerContext,
    params: ImageParams,
    output: str,
    upscale: UpscaleParams,
    source_image: Image.Image,
    strength: float,
) -> None:
    pipe = load_pipeline(OnnxStableDiffusionImg2ImgPipeline,
                         params.model, params.provider, params.scheduler)

    rng = np.random.RandomState(params.seed)

    result = pipe(
        params.prompt,
        generator=rng,
        guidance_scale=params.cfg,
        image=source_image,
        negative_prompt=params.negative_prompt,
        num_inference_steps=params.steps,
        strength=strength,
    )
    image = result.images[0]
    image = run_upscale_correction(
        ctx, StageParams(), params, image, upscale=upscale)

    dest = save_image(ctx, output, image)
    size = Size(*source_image.size)
    save_params(ctx, output, params, size, upscale=upscale)

    del image
    del result
    run_gc()

    logger.info('finished img2img job: %s', dest)


def run_inpaint_pipeline(
    ctx: ServerContext,
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
    stage = StageParams()
    image = upscale_outpaint(
        ctx,
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
        ctx, stage, params, image, upscale=upscale)

    dest = save_image(ctx, output, image)
    save_params(ctx, output, params, size, upscale=upscale, border=border)

    del image
    run_gc()

    logger.info('finished inpaint job: %s', dest)


def run_upscale_pipeline(
    ctx: ServerContext,
    params: ImageParams,
    size: Size,
    output: str,
    upscale: UpscaleParams,
    source_image: Image.Image,
) -> None:
    image = run_upscale_correction(
        ctx, StageParams(), params, source_image, upscale=upscale)

    dest = save_image(ctx, output, image)
    save_params(ctx, output, params, size, upscale=upscale)

    del image
    run_gc()

    logger.info('finished upscale job: %s', dest)
