from logging import getLogger
from os import path
from typing import Any, List, Optional, Tuple

import numpy as np
from onnx import load_model
from transformers import CLIPTokenizer

from ..constants import ONNX_MODEL
from ..convert.diffusion.lora import blend_loras, buffer_external_data_tensors
from ..convert.diffusion.textual_inversion import blend_textual_inversions
from ..diffusers.pipelines.upscale import OnnxStableDiffusionUpscalePipeline
from ..diffusers.utils import expand_prompt
from ..models.meta import NetworkModel
from ..params import DeviceParams
from ..server import ServerContext
from ..utils import run_gc
from .pipelines.controlnet import OnnxStableDiffusionControlNetPipeline
from .pipelines.lpw import OnnxStableDiffusionLongPromptWeightingPipeline
from .pipelines.pix2pix import OnnxStableDiffusionInstructPix2PixPipeline
from .version_safe_diffusers import (
    DDIMScheduler,
    DDPMScheduler,
    DEISMultistepScheduler,
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
    OnnxStableDiffusionInpaintPipeline,
    OnnxStableDiffusionPipeline,
    PNDMScheduler,
    StableDiffusionPipeline,
    UniPCMultistepScheduler,
)

logger = getLogger(__name__)

available_pipelines = {
    "controlnet": OnnxStableDiffusionControlNetPipeline,
    "img2img": OnnxStableDiffusionImg2ImgPipeline,
    "inpaint": OnnxStableDiffusionInpaintPipeline,
    "lpw": OnnxStableDiffusionLongPromptWeightingPipeline,
    "pix2pix": OnnxStableDiffusionInstructPix2PixPipeline,
    "txt2img": OnnxStableDiffusionPipeline,
    "upscale": OnnxStableDiffusionUpscalePipeline,
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

    torch_dtype = server.torch_dtype()
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
        unet_type = "unet"

        # ControlNet component
        if pipeline == "controlnet" and control is not None:
            cnet_path = path.join(server.model_path, "control", f"{control.name}.onnx")
            logger.debug("loading ControlNet weights from %s", cnet_path)
            components["controlnet"] = OnnxRuntimeModel(
                OnnxRuntimeModel.load_model(
                    cnet_path,
                    provider=device.ort_provider(),
                    sess_options=device.sess_options(),
                )
            )

            unet_type = "cnet"

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
                        sess_options=device.sess_options(),
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
            unet = path.join(model, unet_type, ONNX_MODEL)
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

        # make sure a UNet has been loaded
        if "unet" not in components:
            unet = path.join(model, unet_type, ONNX_MODEL)
            logger.debug("loading UNet (%s) from %s", unet_type, unet)
            components["unet"] = OnnxRuntimeModel(
                OnnxRuntimeModel.load_model(
                    unet,
                    provider=device.ort_provider(),
                    sess_options=device.sess_options(),
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


# TODO: does this need to change for fp16 modes?
timestep_dtype = np.float32


class UNetWrapper(object):
    def __init__(
        self,
        server: ServerContext,
        wrapped: OnnxRuntimeModel,
    ):
        self.server = server
        self.wrapped = wrapped

    def __call__(
        self,
        sample: np.ndarray = None,
        timestep: np.ndarray = None,
        encoder_hidden_states: np.ndarray = None,
        **kwargs,
    ):
        global timestep_dtype
        timestep_dtype = timestep.dtype

        logger.trace(
            "UNet parameter types: %s, %s, %s",
            sample.dtype,
            timestep.dtype,
            encoder_hidden_states.dtype,
        )

        if self.prompt_embeds is not None:
            step_index = self.prompt_index % len(self.prompt_embeds)
            logger.trace("multiple prompt embeds found, using step: %s", step_index)
            encoder_hidden_states = self.prompt_embeds[step_index]
            self.prompt_index += 1

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

    def set_prompts(self, prompt_embeds: List[np.ndarray]):
        logger.debug(
            "setting prompt embeds for UNet: %s", [p.shape for p in prompt_embeds]
        )
        self.prompt_embeds = prompt_embeds
        self.prompt_index = 0


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
    pipe.unet = UNetWrapper(server, original_unet)

    if hasattr(pipe, "vae_decoder"):
        original_vae = pipe.vae_decoder
        pipe.vae_decoder = VAEWrapper(server, original_vae)
    elif hasattr(pipe, "vae"):
        pass  # TODO: current wrapper does not work with upscaling VAE
    else:
        logger.debug("no VAE found to patch")
