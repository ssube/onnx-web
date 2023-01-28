from diffusers import (
    OnnxStableDiffusionImg2ImgPipeline,
)
from PIL import Image

from ..diffusion import (
    load_pipeline,
)
from ..params import (
    ImageParams,
    StageParams,
)
from ..utils import (
    ServerContext,
)

import numpy as np


def blend_img2img(
    ctx: ServerContext,
    stage: StageParams,
    params: ImageParams,
    source_image: Image.Image,
    *,
    strength: float,
) -> Image.Image:
    print('generating image using img2img', params.prompt)

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
    output = result.images[0]

    print('final output image size', output.size)
    return output

