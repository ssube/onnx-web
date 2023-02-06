from logging import getLogger
from typing import Optional

import numpy as np
import torch
from diffusers import OnnxStableDiffusionImg2ImgPipeline
from PIL import Image

from ..device_pool import JobContext
from ..diffusion.load import load_pipeline
from ..params import ImageParams, StageParams
from ..utils import ServerContext

logger = getLogger(__name__)


def blend_img2img(
    job: JobContext,
    _server: ServerContext,
    _stage: StageParams,
    params: ImageParams,
    source_image: Image.Image,
    *,
    strength: float,
    prompt: Optional[str] = None,
    **kwargs,
) -> Image.Image:
    prompt = prompt or params.prompt
    logger.info("generating image using img2img, %s steps: %s", params.steps, prompt)

    pipe = load_pipeline(
        OnnxStableDiffusionImg2ImgPipeline,
        params.model,
        params.scheduler,
        job.get_device(),
        params.lpw,
    )
    if params.lpw:
        logger.debug("using LPW pipeline for img2img")
        rng = torch.manual_seed(params.seed)
        result = pipe.img2img(
            prompt,
            generator=rng,
            guidance_scale=params.cfg,
            image=source_image,
            negative_prompt=params.negative_prompt,
            num_inference_steps=params.steps,
            strength=strength,
        )
    else:
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

    logger.info("final output image size: %sx%s", output.width, output.height)
    return output
