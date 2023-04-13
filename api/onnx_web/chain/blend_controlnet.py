from logging import getLogger
from typing import Optional

import numpy as np
import torch
from PIL import Image

from ..diffusers.load import load_pipeline
from ..diffusers.pipelines.controlnet import OnnxStableDiffusionControlNetPipeline
from ..params import ImageParams, StageParams
from ..server import ServerContext
from ..worker import ProgressCallback, WorkerContext

logger = getLogger(__name__)


def blend_controlnet(
    job: WorkerContext,
    server: ServerContext,
    _stage: StageParams,
    params: ImageParams,
    source: Image.Image,
    *,
    callback: Optional[ProgressCallback] = None,
    stage_source: Image.Image,
    **kwargs,
) -> Image.Image:
    params = params.with_args(**kwargs)
    source = stage_source or source
    logger.info(
        "blending image using ControlNet, %s steps: %s", params.steps, params.prompt
    )

    pipe = load_pipeline(
        server,
        "controlnet",
        params.model,
        params.scheduler,
        job.get_device(),
    )

    rng = np.random.RandomState(params.seed)
    result = pipe(
        params.prompt,
        generator=rng,
        guidance_scale=params.cfg,
        image=source,
        negative_prompt=params.negative_prompt,
        num_inference_steps=params.steps,
        strength=params.strength, # TODO: ControlNet strength
        callback=callback,
    )

    output = result.images[0]

    logger.info("final output image size: %sx%s", output.width, output.height)
    return output
