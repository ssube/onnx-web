from logging import getLogger
from os import path
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
from diffusers import (
    DDIMScheduler,
    DDPMScheduler,
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
    OnnxStableDiffusionImg2ImgPipeline,
    OnnxStableDiffusionPipeline,
    PNDMScheduler,
    StableDiffusionPipeline,
)
from onnx import load_model
from transformers import CLIPTokenizer

try:
    from diffusers import DEISMultistepScheduler
except ImportError:
    from ..diffusers.stub_scheduler import StubScheduler as DEISMultistepScheduler

try:
    from diffusers import UniPCMultistepScheduler
except ImportError:
    from ..diffusers.stub_scheduler import StubScheduler as UniPCMultistepScheduler

from ..constants import ONNX_MODEL
from ..convert.diffusion.lora import blend_loras, buffer_external_data_tensors
from ..convert.diffusion.textual_inversion import blend_textual_inversions
from ..diffusers.lpw_stable_diffusion_onnx import (
    OnnxStableDiffusionLongPromptWeightingPipeline,
)
from ..diffusers.pipelines.controlnet import OnnxStableDiffusionControlNetPipeline
from ..diffusers.pipelines.pix2pix import OnnxStableDiffusionInstructPix2PixPipeline
from ..diffusers.utils import expand_prompt
from ..models.meta import NetworkModel
from ..params import DeviceParams, Size
from ..server import ServerContext
from ..utils import run_gc

logger = getLogger(__name__)

latent_channels = 4
latent_factor = 8

available_pipelines = {
    "controlnet": OnnxStableDiffusionControlNetPipeline,
    # "img2img": OnnxStableDiffusionImg2ImgPipeline,
    # "inpaint": OnnxStableDiffusionInpaintPipeline,
    "lpw": OnnxStableDiffusionLongPromptWeightingPipeline,
    "pix2pix": OnnxStableDiffusionInstructPix2PixPipeline,
    # "txt2img": OnnxStableDiffusionPipeline,
}

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


def get_available_pipelines() -> List[str]:
    return list(available_pipelines.keys())


def get_pipeline_schedulers() -> List[str]:
    return list(pipeline_schedulers.keys())


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
    pipeline: str,
    model: str,
    scheduler_name: str,
    device: DeviceParams,
    control: Optional[NetworkModel] = None,
    inversions: Optional[List[Tuple[str, float]]] = None,
    loras: Optional[List[Tuple[str, float]]] = None,
):
    inversions = inversions or []
    loras = loras or []
    control_key = control.name if control is not None else None

    torch_dtype = (
        torch.float16 if "torch-fp16" in server.optimizations else torch.float32
    )
    logger.debug("using Torch dtype %s for pipeline", torch_dtype)
    pipe_key = (
        pipeline,
        model,
        device.device,
        device.provider,
        control_key,
        inversions,
        loras,
    )
    scheduler_key = (scheduler_name, model)
    scheduler_type = pipeline_schedulers[scheduler_name]

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
                torch_dtype=torch_dtype,
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

        logger.debug("loading new diffusion pipeline from %s", model)
        components = {
            "scheduler": scheduler_type.from_pretrained(
                model,
                provider=device.ort_provider(),
                sess_options=device.sess_options(),
                subfolder="scheduler",
                torch_dtype=torch_dtype,
            )
        }

        # shared components
        text_encoder = None
        unet = None

        # ControlNet component
        if pipeline == "controlnet" and control is not None:
            components["controlnet"] = OnnxRuntimeModel(
                OnnxRuntimeModel.load_model(
                    path.join(server.model_path, "control", f"{control.name}.onnx"),
                    provider=device.ort_provider(),
                    sess_options=device.sess_options(),
                )
            )

            unet = path.join(model, "cnet", ONNX_MODEL)

        # Textual Inversion blending
        if inversions is not None and len(inversions) > 0:
            logger.debug("blending Textual Inversions from %s", inversions)
            inversion_names, inversion_weights = zip(*inversions)

            inversion_models = [
                path.join(server.model_path, "inversion", name)
                for name in inversion_names
            ]
            text_encoder = load_model(path.join(model, "text_encoder", ONNX_MODEL))
            tokenizer = CLIPTokenizer.from_pretrained(
                model,
                subfolder="tokenizer",
                torch_dtype=torch_dtype,
            )
            text_encoder, tokenizer = blend_textual_inversions(
                server,
                text_encoder,
                tokenizer,
                list(
                    zip(
                        inversion_models,
                        inversion_weights,
                        inversion_names,
                        [None] * len(inversion_models),
                    )
                ),
            )

            components["tokenizer"] = tokenizer

            # should be pretty small and should not need external data
            if loras is None or len(loras) == 0:
                components["text_encoder"] = OnnxRuntimeModel(
                    OnnxRuntimeModel.load_model(
                        text_encoder.SerializeToString(),
                        provider=device.ort_provider(),
                    )
                )

        # LoRA blending
        if loras is not None and len(loras) > 0:
            lora_names, lora_weights = zip(*loras)
            lora_models = [
                path.join(server.model_path, "lora", name) for name in lora_names
            ]
            logger.info(
                "blending base model %s with LoRA models: %s", model, lora_models
            )

            # blend and load text encoder
            text_encoder = text_encoder or path.join(model, "text_encoder", ONNX_MODEL)
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
            unet = unet or path.join(model, "unet", ONNX_MODEL)
            blended_unet = blend_loras(
                server,
                unet,
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

        pipeline_class = available_pipelines.get(pipeline, OnnxStableDiffusionPipeline)
        logger.debug("loading pretrained SD pipeline for %s", pipeline_class.__name__)
        pipe = pipeline_class.from_pretrained(
            model,
            provider=device.ort_provider(),
            sess_options=device.sess_options(),
            safety_checker=None,
            torch_dtype=torch_dtype,
            **components,
        )

        if not server.show_progress:
            pipe.set_progress_bar_config(disable=True)

        optimize_pipeline(server, pipe)

        # TODO: CPU VAE, etc
        if device is not None and hasattr(pipe, "to"):
            pipe = pipe.to(device.torch_str())

        # monkey-patch pipeline
        patch_pipeline(server, pipe, pipeline)

        server.cache.set("diffusion", pipe_key, pipe)
        server.cache.set("scheduler", scheduler_key, components["scheduler"])

    return pipe


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


timestep_dtype = None


class UNetWrapper(object):
    def __init__(self, server, wrapped):
        self.server = server
        self.wrapped = wrapped

    def __call__(
        self, sample=None, timestep=None, encoder_hidden_states=None, **kwargs
    ):
        global timestep_dtype
        timestep_dtype = timestep.dtype

        logger.trace(
            "UNet parameter types: %s, %s, %s",
            sample.dtype,
            timestep.dtype,
            encoder_hidden_states.dtype,
        )
        if sample.dtype != timestep.dtype:
            logger.trace("converting UNet sample to timestep dtype")
            sample = sample.astype(timestep.dtype)

        if encoder_hidden_states.dtype != timestep.dtype:
            logger.trace("converting UNet hidden states to timestep dtype")
            encoder_hidden_states = encoder_hidden_states.astype(timestep.dtype)

        return self.wrapped(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            **kwargs,
        )

    def __getattr__(self, attr):
        return getattr(self.wrapped, attr)


class VAEWrapper(object):
    def __init__(self, server, wrapped):
        self.server = server
        self.wrapped = wrapped

    def __call__(self, latent_sample=None, **kwargs):
        global timestep_dtype

        logger.trace("VAE parameter types: %s", latent_sample.dtype)
        if latent_sample.dtype != timestep_dtype:
            logger.info("converting VAE sample dtype")
            latent_sample = latent_sample.astype(timestep_dtype)

        return self.wrapped(latent_sample=latent_sample, **kwargs)

    def __getattr__(self, attr):
        return getattr(self.wrapped, attr)


def patch_pipeline(
    server: ServerContext,
    pipe: StableDiffusionPipeline,
    pipeline: Any,
) -> None:
    logger.debug("patching SD pipeline")
    pipe._encode_prompt = expand_prompt.__get__(pipe, pipeline)

    original_unet = pipe.unet
    original_vae = pipe.vae_decoder

    pipe.unet = UNetWrapper(server, original_unet)
    pipe.vae_decoder = VAEWrapper(server, original_vae)
