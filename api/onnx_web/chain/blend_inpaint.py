from diffusers import (
    OnnxStableDiffusionInpaintPipeline,
)
from logging import getLogger
from PIL import Image
from typing import Callable, Tuple

from ..diffusion.load import (
    get_latents_from_seed,
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
    is_debug,
    ServerContext,
)
from .utils import (
    process_tile_grid,
)

import numpy as np

logger = getLogger(__name__)


def blend_inpaint(
    ctx: ServerContext,
    stage: StageParams,
    params: ImageParams,
    source_image: Image.Image,
    *,
    expand: Border,
    mask_image: Image.Image = None,
    fill_color: str = 'white',
    mask_filter: Callable = mask_filter_none,
    noise_source: Callable = noise_source_histogram,
    **kwargs,
) -> Image.Image:
    logger.info('upscaling image by expanding borders', expand)

    if mask_image is None:
        # if no mask was provided, keep the full source image
        mask_image = Image.new('RGB', source_image.size, 'black')

    source_image, mask_image, noise_image, _full_dims = expand_image(
        source_image,
        mask_image,
        expand,
        fill=fill_color,
        noise_source=noise_source,
        mask_filter=mask_filter)

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

        latents = get_latents_from_seed(params.seed, size)
        rng = np.random.RandomState(params.seed)

        result = pipe(
            params.prompt,
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
        return result.images[0]

    output = process_tile_grid(source_image, SizeChart.auto, 1, [outpaint])

    logger.info('final output image size', output.size)
    return output
