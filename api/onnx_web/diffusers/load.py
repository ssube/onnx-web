from logging import getLogger
from os import path
from typing import Any, List, Literal, Optional, Tuple

from onnx import load_model
from optimum.onnxruntime import (  # ORTStableDiffusionXLInpaintPipeline,
    ORTStableDiffusionXLImg2ImgPipeline,
    ORTStableDiffusionXLPipeline,
)
from transformers import CLIPTokenizer

from ..constants import LATENT_FACTOR, ONNX_MODEL
from ..convert.diffusion.lora import blend_loras, buffer_external_data_tensors
from ..convert.diffusion.textual_inversion import blend_textual_inversions
from ..diffusers.pipelines.upscale import OnnxStableDiffusionUpscalePipeline
from ..diffusers.utils import expand_prompt as expand_prompt_onnx_legacy
from ..params import DeviceParams, ImageParams
from ..prompt.compel import expand_prompt as expand_prompt_compel
from ..server import ModelTypes, ServerContext
from ..torch_before_ort import InferenceSession
from ..utils import run_gc
from .patches.scheduler import SchedulerPatch
from .patches.unet import UNetWrapper
from .patches.vae import VAEWrapper
from .pipelines.controlnet import OnnxStableDiffusionControlNetPipeline
from .pipelines.lpw import OnnxStableDiffusionLongPromptWeightingPipeline
from .pipelines.panorama import OnnxStableDiffusionPanoramaPipeline
from .pipelines.panorama_xl import ORTStableDiffusionXLPanoramaPipeline
from .pipelines.pix2pix import OnnxStableDiffusionInstructPix2PixPipeline
from .version_safe_diffusers import (
    DDIMScheduler,
    DDPMScheduler,
    DEISMultistepScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSDEScheduler,
    DPMSolverSinglestepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    IPNDMScheduler,
    KDPM2AncestralDiscreteScheduler,
    KDPM2DiscreteScheduler,
    LCMScheduler,
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
    # "highres": OnnxStableDiffusionHighresPipeline,
    "img2img": OnnxStableDiffusionImg2ImgPipeline,
    "img2img-sdxl": ORTStableDiffusionXLImg2ImgPipeline,
    "inpaint": OnnxStableDiffusionInpaintPipeline,
    # "inpaint-sdxl": ORTStableDiffusionXLInpaintPipeline,
    "lpw": OnnxStableDiffusionLongPromptWeightingPipeline,
    "panorama": OnnxStableDiffusionPanoramaPipeline,
    "panorama-sdxl": ORTStableDiffusionXLPanoramaPipeline,
    "pix2pix": OnnxStableDiffusionInstructPix2PixPipeline,
    "txt2img-sdxl": ORTStableDiffusionXLPipeline,
    "txt2img": OnnxStableDiffusionPipeline,
    "upscale": OnnxStableDiffusionUpscalePipeline,
}

pipeline_schedulers = {
    "ddim": DDIMScheduler,
    "ddpm": DDPMScheduler,
    "deis-multi": DEISMultistepScheduler,
    "dpm-multi": DPMSolverMultistepScheduler,
    "dpm-sde": DPMSolverSDEScheduler,
    "dpm-single": DPMSolverSinglestepScheduler,
    "euler": EulerDiscreteScheduler,
    "euler-a": EulerAncestralDiscreteScheduler,
    "heun": HeunDiscreteScheduler,
    "ipndm": IPNDMScheduler,
    "k-dpm-2-a": KDPM2AncestralDiscreteScheduler,
    "k-dpm-2": KDPM2DiscreteScheduler,
    "lcm": LCMScheduler,
    "lms-discrete": LMSDiscreteScheduler,
    "pndm": PNDMScheduler,
    "unipc-multi": UniPCMultistepScheduler,
}


def add_pipeline(name: str, pipeline: Any) -> bool:
    global available_pipelines

    if name in available_pipelines:
        # TODO: decide if this should be allowed or not
        logger.warning("cannot replace existing pipeline: %s", name)
        return False
    else:
        available_pipelines[name] = pipeline
        return True


def get_available_pipelines() -> List[str]:
    return list(available_pipelines.keys())


def get_pipeline_schedulers() -> List[str]:
    return list(pipeline_schedulers.keys())


def get_scheduler_name(scheduler: Any) -> Optional[str]:
    for k, v in pipeline_schedulers.items():
        if scheduler == v or scheduler == v.__name__:
            return k

    return None


VAE_COMPONENTS = ["vae", "vae_decoder", "vae_encoder"]


def load_pipeline(
    server: ServerContext,
    params: ImageParams,
    pipeline: str,
    device: DeviceParams,
    embeddings: Optional[List[Tuple[str, float]]] = None,
    loras: Optional[List[Tuple[str, float]]] = None,
    model: Optional[str] = None,
):
    embeddings = embeddings or []
    loras = loras or []
    model = model or params.model

    torch_dtype = server.torch_dtype()
    logger.debug("using Torch dtype %s for pipeline", torch_dtype)

    control_key = params.control.name if params.control is not None else None
    pipe_key = (
        pipeline,
        model,
        device.device,
        device.provider,
        control_key,
        embeddings,
        loras,
    )
    scheduler_key = (params.scheduler, model)
    scheduler_type = pipeline_schedulers[params.scheduler]

    cache_pipe = server.cache.get(ModelTypes.diffusion, pipe_key)

    if cache_pipe is not None:
        logger.debug("reusing existing diffusion pipeline")
        pipe = cache_pipe

        # update scheduler
        cache_scheduler = server.cache.get(ModelTypes.scheduler, scheduler_key)
        if cache_scheduler is None:
            logger.debug("loading new diffusion scheduler")
            scheduler = scheduler_type.from_pretrained(
                model,
                provider=device.ort_provider("scheduler"),
                sess_options=device.sess_options(),
                subfolder="scheduler",
                torch_dtype=torch_dtype,
            )

            if device is not None and hasattr(scheduler, "to"):
                scheduler = scheduler.to(device.torch_str())

            pipe.scheduler = scheduler
            server.cache.set(ModelTypes.scheduler, scheduler_key, scheduler)
            run_gc([device])

    else:
        if server.cache.drop("diffusion", pipe_key) > 0:
            logger.debug("unloading previous diffusion pipeline")
            run_gc([device])

        logger.debug("loading new diffusion pipeline from %s", model)
        scheduler = scheduler_type.from_pretrained(
            model,
            provider=device.ort_provider("scheduler"),
            sess_options=device.sess_options(),
            subfolder="scheduler",
            torch_dtype=torch_dtype,
        )
        components = {
            "scheduler": scheduler,
        }

        # shared components
        unet_type = "unet"

        # ControlNet component
        if params.is_control() and params.control is not None:
            logger.debug("loading ControlNet components")
            control_components = load_controlnet(server, device, params)
            components.update(control_components)
            unet_type = "cnet"

        # load various pipeline components
        encoder_components = load_text_encoders(
            server, device, model, embeddings, loras, torch_dtype, params
        )
        components.update(encoder_components)

        unet_components = load_unet(server, device, model, loras, unet_type, params)
        components.update(unet_components)

        vae_components = load_vae(server, device, model, params)
        components.update(vae_components)

        pipeline_class = available_pipelines.get(pipeline, OnnxStableDiffusionPipeline)

        if params.is_xl():
            logger.debug("assembling SDXL pipeline for %s", pipeline_class.__name__)
            pipe = pipeline_class(
                components["vae_decoder_session"],
                components["text_encoder_session"],
                components["unet_session"],
                {
                    "force_zeros_for_empty_prompt": True,
                    "requires_aesthetics_score": False,
                },
                components["tokenizer"],
                scheduler,
                vae_encoder_session=components.get("vae_encoder_session", None),
                text_encoder_2_session=components.get("text_encoder_2_session", None),
                tokenizer_2=components.get("tokenizer_2", None),
                add_watermarker=False,  # not so invisible: https://github.com/ssube/onnx-web/issues/438
            )
        else:
            if params.is_control():
                if "controlnet" not in components or components["controlnet"] is None:
                    raise ValueError("ControlNet is required for control pipelines")

                logger.debug(
                    "assembling SD pipeline for %s with ControlNet",
                    pipeline_class.__name__,
                )
                pipe = pipeline_class(
                    components["vae_encoder"],
                    components["vae_decoder"],
                    components["text_encoder"],
                    components["tokenizer"],
                    components["unet"],
                    components["controlnet"],
                    scheduler,
                    None,
                    None,
                    requires_safety_checker=False,
                )
            elif "vae" in components:
                # upscale uses a single VAE
                logger.debug(
                    "assembling SD pipeline for %s with single VAE",
                    pipeline_class.__name__,
                )
                pipe = pipeline_class(
                    components["vae"],
                    components["text_encoder"],
                    components["tokenizer"],
                    components["unet"],
                    scheduler,
                    scheduler,
                )
            else:
                logger.debug(
                    "assembling SD pipeline for %s with VAE codec",
                    pipeline_class.__name__,
                )
                pipe = pipeline_class(
                    components["vae_encoder"],
                    components["vae_decoder"],
                    components["text_encoder"],
                    components["tokenizer"],
                    components["unet"],
                    scheduler,
                    None,
                    None,
                    requires_safety_checker=False,
                )

        if not server.show_progress:
            pipe.set_progress_bar_config(disable=True)

        optimize_pipeline(server, pipe)
        patch_pipeline(server, pipe, pipeline_class, params)

        server.cache.set(ModelTypes.diffusion, pipe_key, pipe)
        server.cache.set(ModelTypes.scheduler, scheduler_key, scheduler)

    for vae in VAE_COMPONENTS:
        if hasattr(pipe, vae):
            vae_model = getattr(pipe, vae)
            if isinstance(vae_model, VAEWrapper):
                vae_model.set_tiled(tiled=params.tiled_vae)
                vae_model.set_window_size(
                    params.vae_tile // LATENT_FACTOR, params.vae_overlap
                )

    # update panorama params
    if params.is_panorama():
        unet_stride = (params.unet_tile * (1 - params.unet_overlap)) // LATENT_FACTOR
        logger.debug(
            "setting panorama window parameters: %s/%s for UNet, %s/%s for VAE",
            params.unet_tile,
            unet_stride,
            params.vae_tile,
            params.vae_overlap,
        )
        pipe.set_window_size(params.unet_tile // LATENT_FACTOR, unet_stride)

    run_gc([device])

    return pipe


def load_controlnet(server: ServerContext, device: DeviceParams, params: ImageParams):
    cnet_path = path.join(server.model_path, "control", f"{params.control.name}.onnx")
    logger.debug("loading ControlNet weights from %s", cnet_path)
    components = {}
    components["controlnet"] = OnnxRuntimeModel(
        OnnxRuntimeModel.load_model(
            cnet_path,
            provider=device.ort_provider("controlnet"),
            sess_options=device.sess_options(),
        )
    )
    return components


def load_text_encoders(
    server: ServerContext,
    device: DeviceParams,
    model: str,
    embeddings: Optional[List[Tuple[str, float]]],
    loras: Optional[List[Tuple[str, float]]],
    torch_dtype,
    params: ImageParams,
):
    text_encoder = load_model(path.join(model, "text_encoder", ONNX_MODEL))
    tokenizer = CLIPTokenizer.from_pretrained(
        model,
        subfolder="tokenizer",
        torch_dtype=torch_dtype,
    )

    components = {
        "tokenizer": tokenizer,
    }

    if params.is_xl():
        text_encoder_2 = load_model(path.join(model, "text_encoder_2", ONNX_MODEL))
        tokenizer_2 = CLIPTokenizer.from_pretrained(
            model,
            subfolder="tokenizer_2",
            torch_dtype=torch_dtype,
        )
        components["tokenizer_2"] = tokenizer_2

    # blend embeddings, if any
    if embeddings is not None and len(embeddings) > 0:
        embedding_names, embedding_weights = zip(*embeddings)
        embedding_models = [
            path.join(server.model_path, "inversion", name) for name in embedding_names
        ]
        logger.debug(
            "blending base model %s with embeddings from %s", model, embedding_models
        )

        # TODO: blend text_encoder_2 as well
        text_encoder, tokenizer = blend_textual_inversions(
            server,
            text_encoder,
            tokenizer,
            list(
                zip(
                    embedding_models,
                    embedding_weights,
                    embedding_names,
                    [None] * len(embedding_models),
                )
            ),
        )
        components["tokenizer"] = tokenizer

        if params.is_xl():
            text_encoder_2, tokenizer_2 = blend_textual_inversions(
                server,
                text_encoder_2,
                tokenizer_2,
                list(
                    zip(
                        embedding_models,
                        embedding_weights,
                        embedding_names,
                        [None] * len(embedding_models),
                    )
                ),
            )
            components["tokenizer_2"] = tokenizer_2

    # blend LoRAs, if any
    if loras is not None and len(loras) > 0:
        lora_names, lora_weights = zip(*loras)
        lora_models = [
            path.join(server.model_path, "lora", name) for name in lora_names
        ]
        logger.info("blending base model %s with LoRAs from %s", model, lora_models)

        # blend and load text encoder
        text_encoder = blend_loras(
            server,
            text_encoder,
            list(zip(lora_models, lora_weights)),
            "text_encoder",
            1 if params.is_xl() else None,
            params.is_xl(),
        )

        if params.is_xl():
            text_encoder_2 = blend_loras(
                server,
                text_encoder_2,
                list(zip(lora_models, lora_weights)),
                "text_encoder",
                2,
                params.is_xl(),
            )

    # prepare external data for sessions
    (text_encoder, text_encoder_data) = buffer_external_data_tensors(text_encoder)
    text_encoder_names, text_encoder_values = zip(*text_encoder_data)
    text_encoder_opts = device.sess_options(cache=False)
    text_encoder_opts.add_external_initializers(
        list(text_encoder_names), list(text_encoder_values)
    )

    if params.is_xl():
        # encoder 2 only exists in XL
        (text_encoder_2, text_encoder_2_data) = buffer_external_data_tensors(
            text_encoder_2
        )
        text_encoder_2_names, text_encoder_2_values = zip(*text_encoder_2_data)
        text_encoder_2_opts = device.sess_options(cache=False)
        text_encoder_2_opts.add_external_initializers(
            list(text_encoder_2_names), list(text_encoder_2_values)
        )

        # session for te1
        text_encoder_session = InferenceSession(
            text_encoder.SerializeToString(),
            providers=[device.ort_provider("text-encoder", "sdxl")],
            sess_options=text_encoder_opts,
        )
        text_encoder_session._model_path = path.join(model, "text_encoder")
        components["text_encoder_session"] = text_encoder_session

        # session for te2
        text_encoder_2_session = InferenceSession(
            text_encoder_2.SerializeToString(),
            providers=[device.ort_provider("text-encoder", "sdxl")],
            sess_options=text_encoder_2_opts,
        )
        text_encoder_2_session._model_path = path.join(model, "text_encoder_2")
        components["text_encoder_2_session"] = text_encoder_2_session
    else:
        # session for te
        components["text_encoder"] = OnnxRuntimeModel(
            OnnxRuntimeModel.load_model(
                text_encoder.SerializeToString(),
                provider=device.ort_provider("text-encoder"),
                sess_options=text_encoder_opts,
            )
        )

    return components


def load_unet(
    server: ServerContext,
    device: DeviceParams,
    model: str,
    loras: List[Tuple[str, float]],
    unet_type: Literal["cnet", "unet"],
    params: ImageParams,
):
    components = {}
    unet = load_model(path.join(model, unet_type, ONNX_MODEL))

    # LoRA blending
    if loras is not None and len(loras) > 0:
        lora_names, lora_weights = zip(*loras)
        lora_models = [
            path.join(server.model_path, "lora", name) for name in lora_names
        ]
        logger.info("blending base model %s with LoRA models: %s", model, lora_models)

        # blend and load unet
        unet = blend_loras(
            server,
            unet,
            list(zip(lora_models, lora_weights)),
            "unet",
            xl=params.is_xl(),
        )

    (unet_model, unet_data) = buffer_external_data_tensors(unet)
    unet_names, unet_values = zip(*unet_data)
    unet_opts = device.sess_options(cache=False)
    unet_opts.add_external_initializers(list(unet_names), list(unet_values))

    if params.is_xl():
        unet_session = InferenceSession(
            unet_model.SerializeToString(),
            providers=[device.ort_provider("unet", "sdxl")],
            sess_options=unet_opts,
        )
        unet_session._model_path = path.join(model, "unet")
        components["unet_session"] = unet_session
    else:
        components["unet"] = OnnxRuntimeModel(
            OnnxRuntimeModel.load_model(
                unet_model.SerializeToString(),
                provider=device.ort_provider("unet"),
                sess_options=unet_opts,
            )
        )

    return components


def load_vae(
    _server: ServerContext, device: DeviceParams, model: str, params: ImageParams
):
    # one or more VAE models need to be loaded
    vae = path.join(model, "vae", ONNX_MODEL)
    vae_decoder = path.join(model, "vae_decoder", ONNX_MODEL)
    vae_encoder = path.join(model, "vae_encoder", ONNX_MODEL)

    components = {}
    if not params.is_xl() and path.exists(vae):
        logger.debug("loading VAE from %s", vae)
        components["vae"] = OnnxRuntimeModel(
            OnnxRuntimeModel.load_model(
                vae,
                provider=device.ort_provider("vae"),
                sess_options=device.sess_options(),
            )
        )
    elif path.exists(vae_decoder) and path.exists(vae_encoder):
        if params.is_xl():
            logger.debug("loading VAE decoder from %s", vae_decoder)
            components["vae_decoder_session"] = OnnxRuntimeModel.load_model(
                vae_decoder,
                provider=device.ort_provider("vae", "sdxl"),
                sess_options=device.sess_options(),
            )
            components["vae_decoder_session"]._model_path = vae_decoder

            logger.debug("loading VAE encoder from %s", vae_encoder)
            components["vae_encoder_session"] = OnnxRuntimeModel.load_model(
                vae_encoder,
                provider=device.ort_provider("vae", "sdxl"),
                sess_options=device.sess_options(),
            )
            components["vae_encoder_session"]._model_path = vae_encoder

        else:
            logger.debug("loading VAE decoder from %s", vae_decoder)
            components["vae_decoder"] = OnnxRuntimeModel(
                OnnxRuntimeModel.load_model(
                    vae_decoder,
                    provider=device.ort_provider("vae"),
                    sess_options=device.sess_options(),
                )
            )

            logger.debug("loading VAE encoder from %s", vae_encoder)
            components["vae_encoder"] = OnnxRuntimeModel(
                OnnxRuntimeModel.load_model(
                    vae_encoder,
                    provider=device.ort_provider("vae"),
                    sess_options=device.sess_options(),
                )
            )

    return components


def optimize_pipeline(
    server: ServerContext,
    pipe: StableDiffusionPipeline,
) -> None:
    if server.has_optimization(
        "diffusers-attention-slicing"
    ) or server.has_optimization("diffusers-attention-slicing-auto"):
        logger.debug("enabling auto attention slicing on SD pipeline")
        try:
            pipe.enable_attention_slicing(slice_size="auto")
        except Exception as e:
            logger.warning("error while enabling auto attention slicing: %s", e)

    if server.has_optimization("diffusers-attention-slicing-max"):
        logger.debug("enabling max attention slicing on SD pipeline")
        try:
            pipe.enable_attention_slicing(slice_size="max")
        except Exception as e:
            logger.warning("error while enabling max attention slicing: %s", e)

    if server.has_optimization("diffusers-vae-slicing"):
        logger.debug("enabling VAE slicing on SD pipeline")
        try:
            pipe.enable_vae_slicing()
        except Exception as e:
            logger.warning("error while enabling VAE slicing: %s", e)

    if server.has_optimization("diffusers-cpu-offload-sequential"):
        logger.debug("enabling sequential CPU offload on SD pipeline")
        try:
            pipe.enable_sequential_cpu_offload()
        except Exception as e:
            logger.warning("error while enabling sequential CPU offload: %s", e)

    elif server.has_optimization("diffusers-cpu-offload-model"):
        # TODO: check for accelerate
        logger.debug("enabling model CPU offload on SD pipeline")
        try:
            pipe.enable_model_cpu_offload()
        except Exception as e:
            logger.warning("error while enabling model CPU offload: %s", e)

    if server.has_optimization("diffusers-memory-efficient-attention"):
        # TODO: check for xformers
        logger.debug("enabling memory efficient attention for SD pipeline")
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception as e:
            logger.warning("error while enabling memory efficient attention: %s", e)


IMAGE_PIPELINES = [
    OnnxStableDiffusionControlNetPipeline,
    OnnxStableDiffusionImg2ImgPipeline,
    OnnxStableDiffusionInpaintPipeline,
    OnnxStableDiffusionInstructPix2PixPipeline,
    OnnxStableDiffusionLongPromptWeightingPipeline,
    OnnxStableDiffusionPanoramaPipeline,
    OnnxStableDiffusionUpscalePipeline,
    ORTStableDiffusionXLImg2ImgPipeline,
    ORTStableDiffusionXLPanoramaPipeline,
]


def patch_pipeline(
    server: ServerContext,
    pipe: StableDiffusionPipeline,
    pipeline: Any,
    params: ImageParams,
) -> None:
    logger.debug("patching SD pipeline")

    if server.has_feature("compel-prompts"):
        logger.debug("patching prompt encoder with Compel")
        pipe._encode_prompt = expand_prompt_compel.__get__(pipe, pipeline)
    else:
        if not params.is_lpw() and not params.is_xl():
            logger.debug("patching prompt encoder with ONNX legacy method")
            pipe._encode_prompt = expand_prompt_onnx_legacy.__get__(pipe, pipeline)
        else:
            logger.warning("no prompt encoder patch available")

    # the pipeline requested in params may not be the one currently being used, especially during the later img2img
    # stages of a highres pipeline, so we need to check the pipeline type
    is_text_pipeline = type(pipe) not in IMAGE_PIPELINES
    logger.debug(
        "patching pipeline scheduler for %s pipeline",
        "txt2img" if is_text_pipeline else "img2img",
    )
    original_scheduler = pipe.scheduler
    pipe.scheduler = SchedulerPatch(server, original_scheduler, is_text_pipeline)

    logger.debug("patching pipeline UNet")
    original_unet = pipe.unet
    pipe.unet = UNetWrapper(server, original_unet, params.is_xl())

    if hasattr(pipe, "vae_decoder"):
        original_decoder = pipe.vae_decoder
        pipe.vae_decoder = VAEWrapper(
            server,
            original_decoder,
            decoder=True,
            window=params.unet_tile,
            overlap=params.vae_overlap,
        )
        logger.debug("patched VAE decoder with wrapper")

    if hasattr(pipe, "vae_encoder"):
        original_encoder = pipe.vae_encoder
        pipe.vae_encoder = VAEWrapper(
            server,
            original_encoder,
            decoder=False,
            window=params.unet_tile,
            overlap=params.vae_overlap,
        )
        logger.debug("patched VAE encoder with wrapper")

    if hasattr(pipe, "vae"):
        logger.warning("not patching single VAE, tiled VAE may not work")
