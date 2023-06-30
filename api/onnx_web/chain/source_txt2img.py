from logging import getLogger
from typing import Optional

import numpy as np
import torch
from PIL import Image

from ..diffusers.load import load_pipeline
from ..diffusers.utils import encode_prompt, get_latents_from_seed, parse_prompt
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
            "a source image was passed to a txt2img stage, and will be discarded"
        )

    prompt_pairs, loras, inversions = parse_prompt(params)

    latents = get_latents_from_seed(params.seed, size)
    pipe_type = params.get_valid_pipeline("txt2img")
    pipe = load_pipeline(
        server,
        params,
        pipe_type,
        job.get_device(),
        inversions=inversions,
        loras=loras,
    )

    if params.lpw():
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
        # encode and record alternative prompts outside of LPW
        prompt_embeds = encode_prompt(pipe, prompt_pairs, params.batch, params.do_cfg())
        pipe.unet.set_prompts(prompt_embeds)

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
