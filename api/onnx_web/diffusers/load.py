from logging import getLogger
from os import path
from typing import Any, Optional

from diffusers import (
    DDIMScheduler,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    IPNDMScheduler,
    KarrasVeScheduler,
    KDPM2AncestralDiscreteScheduler,
    KDPM2DiscreteScheduler,
    LMSDiscreteScheduler,
    OnnxRuntimeModel,
    PNDMScheduler,
    StableDiffusionPipeline,
)
from transformers import CLIPTokenizer

from .utils import expand_prompt

try:
    from diffusers import DEISMultistepScheduler
except ImportError:
    from ..diffusers.stub_scheduler import StubScheduler as DEISMultistepScheduler

try:
    from diffusers import UniPCMultistepScheduler
except ImportError:
    from ..diffusers.stub_scheduler import StubScheduler as UniPCMultistepScheduler

from ..params import DeviceParams
from ..server import ServerContext
from ..utils import run_gc

logger = getLogger(__name__)

pipeline_schedulers = {
    "ddim": DDIMScheduler,
    "ddpm": DDPMScheduler,
    "deis-multi": DEISMultistepScheduler,
    "dpm-multi": DPMSolverMultistepScheduler,
    "dpm-single": DPMSolverSinglestepScheduler,
    "euler": EulerDiscreteScheduler,
    "euler-a": EulerAncestralDiscreteScheduler,
    "heun": HeunDiscreteScheduler,
    "ipndm": IPNDMScheduler,
    "k-dpm-2-a": KDPM2AncestralDiscreteScheduler,
    "k-dpm-2": KDPM2DiscreteScheduler,
    "karras-ve": KarrasVeScheduler,
    "lms-discrete": LMSDiscreteScheduler,
    "pndm": PNDMScheduler,
    "unipc-multi": UniPCMultistepScheduler,
}


def get_pipeline_schedulers():
    return pipeline_schedulers


def get_scheduler_name(scheduler: Any) -> Optional[str]:
    for k, v in pipeline_schedulers.items():
        if scheduler == v or scheduler == v.__name__:
            return k

    return None


def optimize_pipeline(
    server: ServerContext,
    pipe: StableDiffusionPipeline,
) -> None:
    if "diffusers-attention-slicing" in server.optimizations:
        logger.debug("enabling attention slicing on SD pipeline")
        try:
            pipe.enable_attention_slicing()
        except Exception as e:
            logger.warning("error while enabling attention slicing: %s", e)

    if "diffusers-vae-slicing" in server.optimizations:
        logger.debug("enabling VAE slicing on SD pipeline")
        try:
            pipe.enable_vae_slicing()
        except Exception as e:
            logger.warning("error while enabling VAE slicing: %s", e)

    if "diffusers-cpu-offload-sequential" in server.optimizations:
        logger.debug("enabling sequential CPU offload on SD pipeline")
        try:
            pipe.enable_sequential_cpu_offload()
        except Exception as e:
            logger.warning("error while enabling sequential CPU offload: %s", e)

    elif "diffusers-cpu-offload-model" in server.optimizations:
        # TODO: check for accelerate
        logger.debug("enabling model CPU offload on SD pipeline")
        try:
            pipe.enable_model_cpu_offload()
        except Exception as e:
            logger.warning("error while enabling model CPU offload: %s", e)

    if "diffusers-memory-efficient-attention" in server.optimizations:
        # TODO: check for xformers
        logger.debug("enabling memory efficient attention for SD pipeline")
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception as e:
            logger.warning("error while enabling memory efficient attention: %s", e)


def load_pipeline(
    server: ServerContext,
    pipeline: DiffusionPipeline,
    model: str,
    scheduler_name: str,
    device: DeviceParams,
    lpw: bool,
    inversion: Optional[str],
):
    pipe_key = (
        pipeline.__name__,
        model,
        device.device,
        device.provider,
        lpw,
        inversion,
    )
    scheduler_key = (scheduler_name, model)
    scheduler_type = get_pipeline_schedulers()[scheduler_name]

    cache_pipe = server.cache.get("diffusion", pipe_key)

    if cache_pipe is not None:
        logger.debug("reusing existing diffusion pipeline")
        pipe = cache_pipe

        cache_scheduler = server.cache.get("scheduler", scheduler_key)
        if cache_scheduler is None:
            logger.debug("loading new diffusion scheduler")
            scheduler = scheduler_type.from_pretrained(
                model,
                provider=device.ort_provider(),
                sess_options=device.sess_options(),
                subfolder="scheduler",
            )

            if device is not None and hasattr(scheduler, "to"):
                scheduler = scheduler.to(device.torch_str())

            pipe.scheduler = scheduler
            server.cache.set("scheduler", scheduler_key, scheduler)
            run_gc([device])

    else:
        if server.cache.drop("diffusion", pipe_key) > 0:
            logger.debug("unloading previous diffusion pipeline")
            run_gc([device])

        if lpw:
            custom_pipeline = "./onnx_web/diffusers/lpw_stable_diffusion_onnx.py"
        else:
            custom_pipeline = None

        logger.debug("loading new diffusion pipeline from %s", model)
        components = {
            "scheduler": scheduler_type.from_pretrained(
                model,
                provider=device.ort_provider(),
                sess_options=device.sess_options(),
                subfolder="scheduler",
            )
        }

        if inversion is not None:
            logger.debug("loading text encoder from %s", inversion)
            components["text_encoder"] = OnnxRuntimeModel.from_pretrained(
                path.join(inversion, "text_encoder"),
                provider=device.ort_provider(),
                sess_options=device.sess_options(),
            )
            components["tokenizer"] = CLIPTokenizer.from_pretrained(
                path.join(inversion, "tokenizer"),
            )

        pipe = pipeline.from_pretrained(
            model,
            custom_pipeline=custom_pipeline,
            provider=device.ort_provider(),
            sess_options=device.sess_options(),
            revision="onnx",
            safety_checker=None,
            **components,
        )

        if not server.show_progress:
            pipe.set_progress_bar_config(disable=True)

        optimize_pipeline(server, pipe)

        if device is not None and hasattr(pipe, "to"):
            pipe = pipe.to(device.torch_str())

        # monkey-patch pipeline
        if not lpw:
            pipe._encode_prompt = expand_prompt.__get__(pipe, pipeline)

        server.cache.set("diffusion", pipe_key, pipe)
        server.cache.set("scheduler", scheduler_key, components["scheduler"])

    return pipe
