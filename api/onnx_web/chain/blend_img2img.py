from logging import getLogger

import numpy as np
import torch
from diffusers import OnnxStableDiffusionImg2ImgPipeline
from PIL import Image

from ..diffusion.load import load_pipeline
from ..params import ImageParams, StageParams
from ..server import JobContext, ProgressCallback, ServerContext

logger = getLogger(__name__)


def blend_img2img(
    job: JobContext,
    server: ServerContext,
    _stage: StageParams,
    params: ImageParams,
    source: Image.Image,
    *,
    callback: ProgressCallback = None,
    stage_source: Image.Image,
    **kwargs,
) -> Image.Image:
    params = params.with_args(**kwargs)
    source = stage_source or source
    logger.info(
        "blending image using img2img, %s steps: %s", params.steps, params.prompt
    )

    pipe = load_pipeline(
        server,
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
            params.prompt,
            generator=rng,
            guidance_scale=params.cfg,
            image=source,
            negative_prompt=params.negative_prompt,
            num_inference_steps=params.steps,
            strength=params.strength,
            callback=callback,
        )
    else:
        rng = np.random.RandomState(params.seed)
        result = pipe(
            params.prompt,
            generator=rng,
            guidance_scale=params.cfg,
            image=source,
            negative_prompt=params.negative_prompt,
            num_inference_steps=params.steps,
            strength=params.strength,
            callback=callback,
        )

    output = result.images[0]

    logger.info("final output image size: %sx%s", output.width, output.height)
    return output
