from logging import getLogger
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image

from ..diffusers.load import load_pipeline
from ..diffusers.upscale import append_upscale_correction
from ..diffusers.utils import parse_prompt
from ..params import HighresParams, ImageParams, StageParams, UpscaleParams
from ..server import ServerContext
from ..worker import WorkerContext
from ..worker.context import ProgressCallback

logger = getLogger(__name__)


def upscale_highres(
    job: WorkerContext,
    server: ServerContext,
    _stage: StageParams,
    params: ImageParams,
    source: Image.Image,
    *,
    highres: HighresParams,
    upscale: UpscaleParams,
    stage_source: Optional[Image.Image] = None,
    pipeline: Optional[Any] = None,
    callback: Optional[ProgressCallback] = None,
    **kwargs,
) -> Image.Image:
    image = stage_source or source

    if highres.scale <= 1:
        return image

    # load img2img pipeline once
    pipe_type = params.get_valid_pipeline("img2img")
    logger.debug("using %s pipeline for highres", pipe_type)

    _prompt_pairs, loras, inversions = parse_prompt(params)
    highres_pipe = pipeline or load_pipeline(
        server,
        params,
        pipe_type,
        job.get_device(),
        inversions=inversions,
        loras=loras,
    )

    scaled_size = (source.width * highres.scale, source.height * highres.scale)

    # TODO: upscaling within the same stage prevents tiling from happening and causes OOM
    if highres.method == "bilinear":
        logger.debug("using bilinear interpolation for highres")
        source = source.resize(scaled_size, resample=Image.Resampling.BILINEAR)
    elif highres.method == "lanczos":
        logger.debug("using Lanczos interpolation for highres")
        source = source.resize(scaled_size, resample=Image.Resampling.LANCZOS)
    else:
        logger.debug("using upscaling pipeline for highres")
        upscale = append_upscale_correction(
            StageParams(),
            params,
            upscale=upscale.with_args(
                faces=False,
                scale=highres.scale,
                outscale=highres.scale,
            ),
        )
        source = upscale(
            job,
            server,
            source,
            callback=callback,
        )

    if pipe_type == "lpw":
        rng = torch.manual_seed(params.seed)
        result = highres_pipe.img2img(
            source,
            params.prompt,
            generator=rng,
            guidance_scale=params.cfg,
            negative_prompt=params.negative_prompt,
            num_images_per_prompt=1,
            num_inference_steps=highres.steps,
            strength=highres.strength,
            eta=params.eta,
            callback=callback,
        )
        return result.images[0]
    else:
        rng = np.random.RandomState(params.seed)
        result = highres_pipe(
            params.prompt,
            source,
            generator=rng,
            guidance_scale=params.cfg,
            negative_prompt=params.negative_prompt,
            num_images_per_prompt=1,
            num_inference_steps=highres.steps,
            strength=highres.strength,
            eta=params.eta,
            callback=callback,
        )
        return result.images[0]
