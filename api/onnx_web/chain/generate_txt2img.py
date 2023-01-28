from diffusers import (
    OnnxStableDiffusionPipeline,
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


def generate_txt2img(
    ctx: ServerContext,
    stage: StageParams,
    params: ImageParams,
    source_image: Image.Image,
    *,
    size: Size,
) -> Image:
    print('generating image using txt2img', params.prompt)

    if source_image is not None:
        print('a source image was passed to a txt2img stage, but will be discarded')

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
    output = result.images[0]

    print('final output image size', output.size)
    return output
