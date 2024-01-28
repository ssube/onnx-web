from logging import getLogger
from typing import Dict, Optional, Tuple, Union

from flask import request

from ..diffusers.load import get_available_pipelines, get_pipeline_schedulers
from ..diffusers.utils import random_seed
from ..params import (
    Border,
    DeviceParams,
    HighresParams,
    ImageParams,
    Size,
    UpscaleParams,
)
from ..utils import (
    get_and_clamp_float,
    get_and_clamp_int,
    get_boolean,
    get_from_list,
    get_not_empty,
)
from .context import ServerContext
from .load import (
    get_available_platforms,
    get_config_value,
    get_correction_models,
    get_highres_methods,
    get_network_models,
    get_upscaling_models,
)
from .utils import get_model_path

logger = getLogger(__name__)


def build_device(
    _server: ServerContext,
    data: Dict[str, str],
) -> Optional[DeviceParams]:
    # platform stuff
    device = None
    device_name = data.get("platform")

    if device_name is not None and device_name != "any":
        for platform in get_available_platforms():
            if platform.device == device_name:
                device = platform

    return device


def build_params(
    server: ServerContext,
    default_pipeline: str,
    data: Dict[str, str],
) -> ImageParams:
    # diffusion model
    model = get_not_empty(data, "model", get_config_value("model"))
    model_path = get_model_path(server, model)

    control = None
    control_name = data.get("control")
    for network in get_network_models():
        if network.name == control_name:
            control = network

    # pipeline stuff
    pipeline = get_from_list(
        data, "pipeline", get_available_pipelines(), default_pipeline
    )
    scheduler = get_from_list(data, "scheduler", get_pipeline_schedulers())

    if scheduler is None:
        scheduler = get_config_value("scheduler")

    # prompt does not come from config
    prompt = data.get("prompt", "")
    negative_prompt = data.get("negativePrompt", None)

    if negative_prompt is not None and negative_prompt.strip() == "":
        negative_prompt = None

    # image params
    batch = get_and_clamp_int(
        data,
        "batch",
        get_config_value("batch"),
        get_config_value("batch", "max"),
        get_config_value("batch", "min"),
    )
    cfg = get_and_clamp_float(
        data,
        "cfg",
        get_config_value("cfg"),
        get_config_value("cfg", "max"),
        get_config_value("cfg", "min"),
    )
    eta = get_and_clamp_float(
        data,
        "eta",
        get_config_value("eta"),
        get_config_value("eta", "max"),
        get_config_value("eta", "min"),
    )
    loopback = get_and_clamp_int(
        data,
        "loopback",
        get_config_value("loopback"),
        get_config_value("loopback", "max"),
        get_config_value("loopback", "min"),
    )
    steps = get_and_clamp_int(
        data,
        "steps",
        get_config_value("steps"),
        get_config_value("steps", "max"),
        get_config_value("steps", "min"),
    )
    tiled_vae = get_boolean(data, "tiled_vae", get_config_value("tiled_vae"))
    thumbnail = get_boolean(data, "thumbnail", get_config_value("thumbnail"))
    unet_overlap = get_and_clamp_float(
        data,
        "unet_overlap",
        get_config_value("unet_overlap"),
        get_config_value("unet_overlap", "max"),
        get_config_value("unet_overlap", "min"),
    )
    unet_tile = get_and_clamp_int(
        data,
        "unet_tile",
        get_config_value("unet_tile"),
        get_config_value("unet_tile", "max"),
        get_config_value("unet_tile", "min"),
    )
    vae_overlap = get_and_clamp_float(
        data,
        "vae_overlap",
        get_config_value("vae_overlap"),
        get_config_value("vae_overlap", "max"),
        get_config_value("vae_overlap", "min"),
    )
    vae_tile = get_and_clamp_int(
        data,
        "vae_tile",
        get_config_value("vae_tile"),
        get_config_value("vae_tile", "max"),
        get_config_value("vae_tile", "min"),
    )

    seed = int(data.get("seed", -1))
    if seed == -1:
        seed = random_seed()

    params = ImageParams(
        model_path,
        pipeline,
        scheduler,
        prompt,
        cfg,
        steps,
        seed,
        eta=eta,
        negative_prompt=negative_prompt,
        batch=batch,
        control=control,
        loopback=loopback,
        tiled_vae=tiled_vae,
        unet_overlap=unet_overlap,
        unet_tile=unet_tile,
        vae_overlap=vae_overlap,
        vae_tile=vae_tile,
        thumbnail=thumbnail,
    )

    return params


def build_size(
    _server: ServerContext,
    data: Dict[str, str],
) -> Size:
    height = get_and_clamp_int(
        data,
        "height",
        get_config_value("height"),
        get_config_value("height", "max"),
        get_config_value("height", "min"),
    )
    width = get_and_clamp_int(
        data,
        "width",
        get_config_value("width"),
        get_config_value("width", "max"),
        get_config_value("width", "min"),
    )
    return Size(width, height)


def build_border(
    data: Dict[str, str] = None,
) -> Border:
    if data is None:
        data = request.args

    left = get_and_clamp_int(
        data,
        "left",
        get_config_value("left"),
        get_config_value("left", "max"),
        get_config_value("left", "min"),
    )
    right = get_and_clamp_int(
        data,
        "right",
        get_config_value("right"),
        get_config_value("right", "max"),
        get_config_value("right", "min"),
    )
    top = get_and_clamp_int(
        data,
        "top",
        get_config_value("top"),
        get_config_value("top", "max"),
        get_config_value("top", "min"),
    )
    bottom = get_and_clamp_int(
        data,
        "bottom",
        get_config_value("bottom"),
        get_config_value("bottom", "max"),
        get_config_value("bottom", "min"),
    )

    return Border(left, right, top, bottom)


def build_upscale(
    data: Dict[str, str] = None,
) -> UpscaleParams:
    if data is None:
        data = request.args

    upscale = get_boolean(data, "upscale", False)
    denoise = get_and_clamp_float(
        data,
        "denoise",
        get_config_value("denoise"),
        get_config_value("denoise", "max"),
        get_config_value("denoise", "min"),
    )
    scale = get_and_clamp_int(
        data,
        "scale",
        get_config_value("scale"),
        get_config_value("scale", "max"),
        get_config_value("scale", "min"),
    )
    outscale = get_and_clamp_int(
        data,
        "outscale",
        get_config_value("outscale"),
        get_config_value("outscale", "max"),
        get_config_value("outscale", "min"),
    )
    upscaling = get_from_list(data, "upscaling", get_upscaling_models())
    correction = get_from_list(data, "correction", get_correction_models())

    faces = get_boolean(data, "faces", False)
    face_outscale = get_and_clamp_int(
        data,
        "faceOutscale",
        get_config_value("faceOutscale"),
        get_config_value("faceOutscale", "max"),
        get_config_value("faceOutscale", "min"),
    )
    face_strength = get_and_clamp_float(
        data,
        "faceStrength",
        get_config_value("faceStrength"),
        get_config_value("faceStrength", "max"),
        get_config_value("faceStrength", "min"),
    )
    upscale_order = data.get("upscaleOrder", "correction-first")

    return UpscaleParams(
        upscaling,
        correction_model=correction,
        denoise=denoise,
        upscale=upscale,
        faces=faces,
        face_outscale=face_outscale,
        face_strength=face_strength,
        outscale=outscale,
        scale=scale,
        upscale_order=upscale_order,
    )


def build_highres(
    data: Dict[str, str] = None,
) -> HighresParams:
    if data is None:
        data = request.args

    enabled = get_boolean(data, "highres", get_config_value("highres"))
    iterations = get_and_clamp_int(
        data,
        "highresIterations",
        get_config_value("highresIterations"),
        get_config_value("highresIterations", "max"),
        get_config_value("highresIterations", "min"),
    )
    method = get_from_list(data, "highresMethod", get_highres_methods())
    scale = get_and_clamp_int(
        data,
        "highresScale",
        get_config_value("highresScale"),
        get_config_value("highresScale", "max"),
        get_config_value("highresScale", "min"),
    )
    steps = get_and_clamp_int(
        data,
        "highresSteps",
        get_config_value("highresSteps"),
        get_config_value("highresSteps", "max"),
        get_config_value("highresSteps", "min"),
    )
    strength = get_and_clamp_float(
        data,
        "highresStrength",
        get_config_value("highresStrength"),
        get_config_value("highresStrength", "max"),
        get_config_value("highresStrength", "min"),
    )

    return HighresParams(
        enabled,
        scale,
        steps,
        strength,
        method=method,
        iterations=iterations,
    )


PipelineParams = Tuple[Optional[DeviceParams], ImageParams, Size]


def pipeline_from_json(
    server: ServerContext,
    data: Dict[str, Union[str, Dict[str, str]]],
    default_pipeline: str = "txt2img",
) -> PipelineParams:
    """
    Like pipeline_from_request but expects a nested structure.
    """

    device = build_device(server, data.get("device", data))
    params = build_params(server, default_pipeline, data.get("params", data))
    size = build_size(server, data.get("params", data))

    return (device, params, size)


def pipeline_from_request(
    server: ServerContext,
    default_pipeline: str = "txt2img",
) -> PipelineParams:
    user = request.remote_addr

    device = build_device(server, request.args)
    params = build_params(server, default_pipeline, request.args)
    size = build_size(server, request.args)

    logger.info(
        "request from %s: %s steps of %s using %s in %s on %s, %sx%s, %s, %s - %s",
        user,
        params.steps,
        params.scheduler,
        params.model,
        params.pipeline,
        device or "any device",
        size.width,
        size.height,
        params.cfg,
        params.seed,
        params.prompt,
    )

    return (device, params, size)
