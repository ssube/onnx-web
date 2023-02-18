from logging import getLogger

import numpy as np
import torch
from diffusers import OnnxStableDiffusionPipeline
from PIL import Image

from ..diffusion.load import get_latents_from_seed, load_pipeline
from ..params import ImageParams, Size, StageParams
from ..server.device_pool import JobContext, ProgressCallback
from ..utils import ServerContext

logger = getLogger(__name__)


def source_txt2img(
    job: JobContext,
    server: ServerContext,
    _stage: StageParams,
    params: ImageParams,
    source: Image.Image,
    *,
    size: Size,
    callback: ProgressCallback = None,
    **kwargs,
) -> Image.Image:
    params = params.with_args(**kwargs)
    size = size.with_args(**kwargs)
    logger.info(
        "generating image using txt2img, %s steps: %s", params.steps, params.prompt
    )

    if source is not None:
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
