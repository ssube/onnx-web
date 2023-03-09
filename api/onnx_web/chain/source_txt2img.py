from logging import getLogger
from typing import Optional

import numpy as np
import torch
from diffusers import OnnxStableDiffusionPipeline
from PIL import Image

from ..diffusers.load import get_latents_from_seed, load_pipeline
from ..params import ImageParams, Size, StageParams
from ..server import ServerContext
from ..worker import ProgressCallback, WorkerContext

logger = getLogger(__name__)


def source_txt2img(
    job: WorkerContext,
    server: ServerContext,
    _stage: StageParams,
    params: ImageParams,
    _source: Image.Image,
    *,
    size: Size,
    callback: Optional[ProgressCallback] = None,
    **kwargs,
) -> Image.Image:
    params = params.with_args(**kwargs)
    size = size.with_args(**kwargs)
    logger.info(
        "generating image using txt2img, %s steps: %s", params.steps, params.prompt
    )

    if "stage_source" in kwargs:
        logger.warn(
            "a source image was passed to a txt2img stage, but will be discarded"
        )

    latents = get_latents_from_seed(params.seed, size)
    pipe = load_pipeline(
        server,
        OnnxStableDiffusionPipeline,
        params.model,
        params.scheduler,
        job.get_device(),
        params.lpw,
        params.inversion,
    )

    if params.lpw:
        logger.debug("using LPW pipeline for txt2img")
        rng = torch.manual_seed(params.seed)
        result = pipe.text2img(
            params.prompt,
            height=size.height,
            width=size.width,
            generator=rng,
            guidance_scale=params.cfg,
            latents=latents,
            negative_prompt=params.negative_prompt,
            num_inference_steps=params.steps,
            callback=callback,
        )
    else:
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
            callback=callback,
        )

    output = result.images[0]

    logger.info("final output image size: %sx%s", output.width, output.height)
    return output
