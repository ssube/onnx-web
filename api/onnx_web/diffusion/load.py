from logging import getLogger
from typing import Any, Optional, Tuple

import numpy as np
from diffusers import (
    DDIMScheduler,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    KarrasVeScheduler,
    KDPM2AncestralDiscreteScheduler,
    KDPM2DiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)

try:
    from diffusers import DEISMultistepScheduler
except:
    from .stub_scheduler import StubScheduler as DEISMultistepScheduler

from ..params import DeviceParams, Size
from ..utils import ServerContext, run_gc

logger = getLogger(__name__)

latent_channels = 4
latent_factor = 8

pipeline_schedulers = {
    "ddim": DDIMScheduler,
    "ddpm": DDPMScheduler,
    "deis-multi": DEISMultistepScheduler,
    "dpm-multi": DPMSolverMultistepScheduler,
    "dpm-single": DPMSolverSinglestepScheduler,
    "euler": EulerDiscreteScheduler,
    "euler-a": EulerAncestralDiscreteScheduler,
    "heun": HeunDiscreteScheduler,
    "k-dpm-2-a": KDPM2AncestralDiscreteScheduler,
    "k-dpm-2": KDPM2DiscreteScheduler,
    "karras-ve": KarrasVeScheduler,
    "lms-discrete": LMSDiscreteScheduler,
    "pndm": PNDMScheduler,
}


def get_scheduler_name(scheduler: Any) -> Optional[str]:
    for k, v in pipeline_schedulers.items():
        if scheduler == v or scheduler == v.__name__:
            return k

    return None


def get_latents_from_seed(seed: int, size: Size, batch: int = 1) -> np.ndarray:
    """
    From https://www.travelneil.com/stable-diffusion-updates.html.
    This one needs to use np.random because of the return type.
    """
    latents_shape = (
        batch,
        latent_channels,
        size.height // latent_factor,
        size.width // latent_factor,
    )
    rng = np.random.default_rng(seed)
    image_latents = rng.standard_normal(latents_shape).astype(np.float32)
    return image_latents


def get_tile_latents(
    full_latents: np.ndarray, dims: Tuple[int, int, int]
) -> np.ndarray:
    x, y, tile = dims
    t = tile // latent_factor
    x = x // latent_factor
    y = y // latent_factor
    xt = x + t
    yt = y + t

    return full_latents[:, :, y:yt, x:xt]


def load_pipeline(
    server: ServerContext,
    pipeline: DiffusionPipeline,
    model: str,
    scheduler_type: Any,
    device: DeviceParams,
    lpw: bool,
):
    pipe_key = (pipeline, model, device.device, device.provider, lpw)
    scheduler_key = (scheduler_type,)

    cache_pipe = server.cache.get("diffusion", pipe_key)

    if cache_pipe is not None:
        logger.debug("reusing existing diffusion pipeline")
        pipe = cache_pipe

        cache_scheduler = server.cache.get("scheduler", scheduler_key)
        if cache_scheduler is None:
            logger.debug("loading new diffusion scheduler")
            scheduler = scheduler_type.from_pretrained(
                model,
                provider=device.provider,
                provider_options=device.options,
                subfolder="scheduler",
            )

            if device is not None and hasattr(scheduler, "to"):
                scheduler = scheduler.to(device.torch_device())

            pipe.scheduler = scheduler
            server.cache.set("scheduler", scheduler_key, scheduler)
            run_gc()

    else:
        logger.debug("unloading previous diffusion pipeline")
        server.cache.drop("diffusion", pipe_key)
        run_gc()

        if lpw:
            custom_pipeline = "./onnx_web/diffusion/lpw_stable_diffusion_onnx.py"
        else:
            custom_pipeline = None

        logger.debug("loading new diffusion pipeline from %s", model)
        scheduler = scheduler_type.from_pretrained(
            model,
            provider=device.provider,
            provider_options=device.options,
            subfolder="scheduler",
        )
        pipe = pipeline.from_pretrained(
            model,
            custom_pipeline=custom_pipeline,
            provider=device.provider,
            provider_options=device.options,
            revision="onnx",
            safety_checker=None,
            scheduler=scheduler,
        )

        if device is not None and hasattr(pipe, "to"):
            pipe = pipe.to(device.torch_device())

        server.cache.set("diffusion", pipe_key, pipe)
        server.cache.set("scheduler", scheduler_key, scheduler)

    return pipe
