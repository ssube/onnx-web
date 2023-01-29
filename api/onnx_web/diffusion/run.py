from diffusers import (
    OnnxStableDiffusionPipeline,
    OnnxStableDiffusionImg2ImgPipeline,
    OnnxStableDiffusionInpaintPipeline,
)
from logging import getLogger
from PIL import Image, ImageChops
from typing import Any

from ..image import (
    expand_image,
)
from ..params import (
    ImageParams,
    Border,
    Size,
    StageParams,
)
from ..upscale import (
    run_upscale_correction,
    UpscaleParams,
)
from ..utils import (
    is_debug,
    base_join,
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

    dest = base_join(ctx.output_path, output)
    image.save(dest)

    del image
    del result

    logger.info('saved txt2img output: %s', dest)


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

    dest = base_join(ctx.output_path, output)
    image.save(dest)

    del image
    del result

    logger.info('saved img2img output: %s', dest)


def run_inpaint_pipeline(
    ctx: ServerContext,
    stage: StageParams,
    params: ImageParams,
    size: Size,
    output: str,
    upscale: UpscaleParams,
    source_image: Image.Image,
    mask_image: Image.Image,
    expand: Border,
    noise_source: Any,
    mask_filter: Any,
    strength: float,
    fill_color: str,
) -> None:
    pipe = load_pipeline(OnnxStableDiffusionInpaintPipeline,
                         params.model, params.provider, params.scheduler)

    latents = get_latents_from_seed(params.seed, size)
    rng = np.random.RandomState(params.seed)

    logger.info('applying mask filter and generating noise source')
    source_image, mask_image, noise_image, _full_dims = expand_image(
        source_image,
        mask_image,
        expand,
        fill=fill_color,
        noise_source=noise_source,
        mask_filter=mask_filter)

    if is_debug():
        source_image.save(base_join(ctx.output_path, 'last-source.png'))
        mask_image.save(base_join(ctx.output_path, 'last-mask.png'))
        noise_image.save(base_join(ctx.output_path, 'last-noise.png'))

    result = pipe(
        params.prompt,
        generator=rng,
        guidance_scale=params.cfg,
        height=size.height,
        image=source_image,
        latents=latents,
        mask_image=mask_image,
        negative_prompt=params.negative_prompt,
        num_inference_steps=params.steps,
        width=size.width,
    )
    image = result.images[0]

    if image.size == source_image.size:
        image = ImageChops.blend(source_image, image, strength)
    else:
        logger.info('output image size does not match source, skipping post-blend')

    image = run_upscale_correction(
        ctx, StageParams(), params, image, upscale=upscale)

    dest = base_join(ctx.output_path, output)
    image.save(dest)

    del image
    del result

    logger.info('saved inpaint output: %s', dest)


def run_upscale_pipeline(
    ctx: ServerContext,
    params: ImageParams,
    _size: Size,
    output: str,
    upscale: UpscaleParams,
    source_image: Image.Image,
) -> None:
    image = run_upscale_correction(
        ctx, StageParams(), params, source_image, upscale=upscale)

    dest = base_join(ctx.output_path, output)
    image.save(dest)

    del image

    logger.info('saved img2img output: %s', dest)
