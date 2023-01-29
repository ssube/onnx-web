from diffusers import (
    OnnxStableDiffusionImg2ImgPipeline,
)
from logging import getLogger
from PIL import Image

from ..diffusion.load import (
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

logger = getLogger(__name__)


def blend_img2img(
    _ctx: ServerContext,
    _stage: StageParams,
    params: ImageParams,
    source_image: Image.Image,
    *,
    strength: float,
    prompt: str = None,
    **kwargs,
) -> Image.Image:
    prompt = prompt or params.prompt
    logger.info('generating image using img2img, %s steps: %s', params.steps, prompt)

    pipe = load_pipeline(OnnxStableDiffusionImg2ImgPipeline,
                            params.model, params.provider, params.scheduler)

    rng = np.random.RandomState(params.seed)

    result = pipe(
        prompt,
        generator=rng,
        guidance_scale=params.cfg,
        image=source_image,
        negative_prompt=params.negative_prompt,
        num_inference_steps=params.steps,
        strength=strength,
    )
    output = result.images[0]

    logger.info('final output image size: %sx%s', output.width, output.height)
    return output

