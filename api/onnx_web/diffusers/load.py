from logging import getLogger
from os import path
from typing import Any, List, Optional, Tuple

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
    IPNDMScheduler,
    KarrasVeScheduler,
    KDPM2AncestralDiscreteScheduler,
    KDPM2DiscreteScheduler,
    LMSDiscreteScheduler,
    OnnxRuntimeModel,
    PNDMScheduler,
    StableDiffusionPipeline,
)
from onnx import load_model
from transformers import CLIPTokenizer

from onnx_web.diffusers.utils import expand_prompt

try:
    from diffusers import DEISMultistepScheduler
except ImportError:
    from ..diffusers.stub_scheduler import StubScheduler as DEISMultistepScheduler

try:
    from diffusers import UniPCMultistepScheduler
except ImportError:
    from ..diffusers.stub_scheduler import StubScheduler as UniPCMultistepScheduler

from ..convert.diffusion.lora import blend_loras, buffer_external_data_tensors
from ..convert.diffusion.textual_inversion import blend_textual_inversions
from ..params import DeviceParams, Size
from ..server import ServerContext
from ..utils import run_gc

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
    inversions: Optional[List[Tuple[str, float]]] = None,
    loras: Optional[List[Tuple[str, float]]] = None,
):
    loras = loras or []
    pipe_key = (
        pipeline.__name__,
        model,
        device.device,
        device.provider,
        lpw,
        inversions,
        loras,
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

        text_encoder = None
        if inversions is not None and len(inversions) > 0:
            inversion_names, inversion_weights = zip(*inversions)
            logger.debug("blending Textual Inversions from %s", inversion_names)

            inversion_models = [
                path.join(server.model_path, "inversion", f"{name}.ckpt")
                for name in inversion_names
            ]
            text_encoder = load_model(path.join(model, "text_encoder", "model.onnx"))
            tokenizer = CLIPTokenizer.from_pretrained(
                model,
                subfolder="tokenizer",
            )
            text_encoder, tokenizer = blend_textual_inversions(
                server,
                text_encoder,
                tokenizer,
                list(zip(inversion_models, inversion_weights, inversion_names)),
            )

            # should be pretty small and should not need external data
            components["text_encoder"] = OnnxRuntimeModel(
                OnnxRuntimeModel.load_model(
                    text_encoder.SerializeToString(),
                    provider=device.ort_provider(),
                )
            )
            components["tokenizer"] = tokenizer

        # test LoRA blending
        if loras is not None and len(loras) > 0:
            lora_names, lora_weights = zip(*loras)
            lora_models = [
                path.join(server.model_path, "lora", f"{name}.safetensors")
                for name in lora_names
            ]
            logger.info(
                "blending base model %s with LoRA models: %s", model, lora_models
            )

            # blend and load text encoder
            text_encoder = text_encoder or path.join(
                model, "text_encoder", "model.onnx"
            )
            text_encoder = blend_loras(
                server,
                text_encoder,
                list(zip(lora_models, lora_weights)),
                "text_encoder",
            )
            (text_encoder, text_encoder_data) = buffer_external_data_tensors(
                text_encoder
            )
            text_encoder_names, text_encoder_values = zip(*text_encoder_data)
            text_encoder_opts = device.sess_options(cache=False)
            text_encoder_opts.add_external_initializers(
                list(text_encoder_names), list(text_encoder_values)
            )
            components["text_encoder"] = OnnxRuntimeModel(
                OnnxRuntimeModel.load_model(
                    text_encoder.SerializeToString(),
                    provider=device.ort_provider(),
                    sess_options=text_encoder_opts,
                )
            )

            # blend and load unet
            blended_unet = blend_loras(
                server,
                path.join(model, "unet", "model.onnx"),
                list(zip(lora_models, lora_weights)),
                "unet",
            )
            (unet_model, unet_data) = buffer_external_data_tensors(blended_unet)
            unet_names, unet_values = zip(*unet_data)
            unet_opts = device.sess_options(cache=False)
            unet_opts.add_external_initializers(list(unet_names), list(unet_values))
            components["unet"] = OnnxRuntimeModel(
                OnnxRuntimeModel.load_model(
                    unet_model.SerializeToString(),
                    provider=device.ort_provider(),
                    sess_options=unet_opts,
                )
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
