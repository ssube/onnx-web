from diffusers import (
    DiffusionPipeline,
)
from logging import getLogger
from typing import Any, Optional, Tuple

from ..params import (
    Size,
)
from ..utils import (
    run_gc,
)

import numpy as np

logger = getLogger(__name__)

last_pipeline_instance = None
last_pipeline_options = (None, None, None)
last_pipeline_scheduler = None


def get_latents_from_seed(seed: int, size: Size) -> np.ndarray:
    '''
    From https://www.travelneil.com/stable-diffusion-updates.html
    '''
    # 1 is batch size
    latents_shape = (1, 4, size.height // 8, size.width // 8)
    # Gotta use numpy instead of torch, because torch's randn() doesn't support DML
    rng = np.random.default_rng(seed)
    image_latents = rng.standard_normal(latents_shape).astype(np.float32)
    return image_latents


def get_tile_latents(full_latents: np.ndarray, dims: Tuple[int, int, int]) -> np.ndarray:
    x, y, tile = dims
    t = tile // 8
    x = x // 8
    y = y // 8
    xt = x + t
    yt = y + t

    return full_latents[:, :, y:yt, x:xt]


def load_pipeline(pipeline: DiffusionPipeline, model: str, provider: str, scheduler: Any, device: Optional[str] = None):
    global last_pipeline_instance
    global last_pipeline_scheduler
    global last_pipeline_options

    options = (pipeline, model, provider)
    if last_pipeline_instance != None and last_pipeline_options == options:
        logger.info('reusing existing diffusion pipeline')
        pipe = last_pipeline_instance
    else:
        logger.info('unloading previous diffusion pipeline')
        last_pipeline_instance = None
        last_pipeline_scheduler = None
        run_gc()

        logger.info('loading new diffusion pipeline')
        pipe = pipeline.from_pretrained(
            model,
            provider=provider,
            safety_checker=None,
            scheduler=scheduler.from_pretrained(model, subfolder='scheduler')
        )

        if device is not None:
            pipe = pipe.to(device)

        last_pipeline_instance = pipe
        last_pipeline_options = options
        last_pipeline_scheduler = scheduler

    if last_pipeline_scheduler != scheduler:
        logger.info('loading new diffusion scheduler')
        scheduler = scheduler.from_pretrained(
            model, subfolder='scheduler')

        if device is not None:
            scheduler = scheduler.to(device)

        pipe.scheduler = scheduler
        last_pipeline_scheduler = scheduler
        run_gc()

    return pipe
