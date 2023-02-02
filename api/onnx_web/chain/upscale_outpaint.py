from diffusers import (
    OnnxStableDiffusionInpaintPipeline,
)
from logging import getLogger
from PIL import Image, ImageDraw
from typing import Callable, Tuple

from ..diffusion.load import (
    get_latents_from_seed,
    get_tile_latents,
    load_pipeline,
)
from ..image import (
    expand_image,
    mask_filter_none,
    noise_source_histogram,
)
from ..params import (
    Border,
    ImageParams,
    Size,
    SizeChart,
    StageParams,
)
from ..output import (
    save_image,
)
from ..utils import (
    base_join,
    is_debug,
    ServerContext,
)
from .utils import (
    process_tile_spiral,
)

import numpy as np

logger = getLogger(__name__)


def upscale_outpaint(
    ctx: ServerContext,
    stage: StageParams,
    params: ImageParams,
    source_image: Image.Image,
    *,
    border: Border,
    prompt: str = None,
    mask_image: Image.Image = None,
    fill_color: str = 'white',
    mask_filter: Callable = mask_filter_none,
    noise_source: Callable = noise_source_histogram,
    **kwargs,
) -> Image.Image:
    prompt = prompt or params.prompt
    logger.info('upscaling image by expanding borders: %s', border)

    if mask_image is None:
        # if no mask was provided, keep the full source image
        mask_image = Image.new('RGB', source_image.size, 'black')

    source_image, mask_image, noise_image, full_dims = expand_image(
        source_image,
        mask_image,
        border,
        fill=fill_color,
        noise_source=noise_source,
        mask_filter=mask_filter)

    draw_mask = ImageDraw.Draw(mask_image)
    full_size = Size(*full_dims)
    full_latents = get_latents_from_seed(params.seed, full_size)

    if is_debug():
        save_image(ctx, 'last-source.png', source_image)
        save_image(ctx, 'last-mask.png', mask_image)
        save_image(ctx, 'last-noise.png', noise_image)

    def outpaint(image: Image.Image, dims: Tuple[int, int, int]):
        left, top, tile = dims
        size = Size(*image.size)
        mask = mask_image.crop((left, top, left + tile, top + tile))

        if is_debug():
            save_image(ctx, 'tile-source.png', image)
            save_image(ctx, 'tile-mask.png', mask)

        # TODO: must use inpainting model here
        model = '../models/stable-diffusion-onnx-v1-inpainting'
        pipe = load_pipeline(OnnxStableDiffusionInpaintPipeline,
                             model, params.provider, params.scheduler)

        latents = get_tile_latents(full_latents, dims)
        rng = np.random.RandomState(params.seed)

        result = pipe(
            prompt,
            generator=rng,
            guidance_scale=params.cfg,
            height=size.height,
            image=image,
            latents=latents,
            mask_image=mask,
            negative_prompt=params.negative_prompt,
            num_inference_steps=params.steps,
            width=size.width,
        )

        # once part of the image has been drawn, keep it
        draw_mask.rectangle((left, top, left + tile, top + tile), fill='black')
        return result.images[0]

    output = process_tile_spiral(source_image, SizeChart.auto, 1, [outpaint])

    logger.info('final output image size: %sx%s', output.width, output.height)
    return output
