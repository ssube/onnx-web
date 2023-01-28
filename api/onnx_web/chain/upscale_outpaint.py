from diffusers import (
    OnnxStableDiffusionInpaintPipeline,
)
from PIL import Image
from typing import Callable

from ..diffusion import (
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
    StageParams,
)
from ..utils import (
    base_join,
    is_debug,
    ServerContext,
)
from .utils import (
    process_tiles,
)

import numpy as np


def upscale_outpaint(
    ctx: ServerContext,
    stage: StageParams,
    params: ImageParams,
    source_image: Image.Image,
    *,
    expand: Border,
    mask_image: Image.Image,
    fill_color: str = 'white',
    mask_filter: Callable = mask_filter_none,
    noise_source: Callable = noise_source_histogram,
) -> Image:
    print('upscaling image by expanding borders', expand)

    output = expand_image(source_image, mask_image, expand)
    size = Size(*output.size)

    def outpaint():
        pipe = load_pipeline(OnnxStableDiffusionInpaintPipeline,
                         params.model, params.provider, params.scheduler)

        latents = get_latents_from_seed(params.seed, size)
        rng = np.random.RandomState(params.seed)

        print('applying mask filter and generating noise source')
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
        return result.images[0]

    output = process_tiles(output, 256, 4, [outpaint])

    print('final output image size', output.size)
    return output
