from logging import getLogger
from typing import Tuple

import numpy as np
from flask import request

from ..diffusers.load import get_available_pipelines, get_pipeline_schedulers
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


def pipeline_from_request(
    server: ServerContext,
    default_pipeline: str = "txt2img",
) -> Tuple[DeviceParams, ImageParams, Size]:
    user = request.remote_addr

    # platform stuff
    device = None
    device_name = request.args.get("platform")

    if device_name is not None and device_name != "any":
        for platform in get_available_platforms():
            if platform.device == device_name:
                device = platform

    # diffusion model
    model = get_not_empty(request.args, "model", get_config_value("model"))
    model_path = get_model_path(server, model)

    control = None
    control_name = request.args.get("control")
    for network in get_network_models():
        if network.name == control_name:
            control = network

    # pipeline stuff
    pipeline = get_from_list(
        request.args, "pipeline", get_available_pipelines(), default_pipeline
    )
    scheduler = get_from_list(request.args, "scheduler", get_pipeline_schedulers())

    if scheduler is None:
        scheduler = get_config_value("scheduler")

    # prompt does not come from config
    prompt = request.args.get("prompt", "")
    negative_prompt = request.args.get("negativePrompt", None)

    if negative_prompt is not None and negative_prompt.strip() == "":
        negative_prompt = None

    # image params
    batch = get_and_clamp_int(
        request.args,
        "batch",
        get_config_value("batch"),
        get_config_value("batch", "max"),
        get_config_value("batch", "min"),
    )
    cfg = get_and_clamp_float(
        request.args,
        "cfg",
        get_config_value("cfg"),
        get_config_value("cfg", "max"),
        get_config_value("cfg", "min"),
    )
    eta = get_and_clamp_float(
        request.args,
        "eta",
        get_config_value("eta"),
        get_config_value("eta", "max"),
        get_config_value("eta", "min"),
    )
    loopback = get_and_clamp_int(
        request.args,
        "loopback",
        get_config_value("loopback"),
        get_config_value("loopback", "max"),
        get_config_value("loopback", "min"),
    )
    steps = get_and_clamp_int(
        request.args,
        "steps",
        get_config_value("steps"),
        get_config_value("steps", "max"),
        get_config_value("steps", "min"),
    )
    height = get_and_clamp_int(
        request.args,
        "height",
        get_config_value("height"),
        get_config_value("height", "max"),
        get_config_value("height", "min"),
    )
    width = get_and_clamp_int(
        request.args,
        "width",
        get_config_value("width"),
        get_config_value("width", "max"),
        get_config_value("width", "min"),
    )
    tiled_vae = get_boolean(request.args, "tiledVAE", get_config_value("tiledVAE"))
    tiles = get_and_clamp_int(
        request.args,
        "tiles",
        get_config_value("tiles"),
        get_config_value("tiles", "max"),
        get_config_value("tiles", "min"),
    )
    overlap = get_and_clamp_float(
        request.args,
        "overlap",
        get_config_value("overlap"),
        get_config_value("overlap", "max"),
        get_config_value("overlap", "min"),
    )
    stride = get_and_clamp_float(
        request.args,
        "stride",
        get_config_value("stride"),
        get_config_value("stride", "max"),
        get_config_value("stride", "min"),
    )

    seed = int(request.args.get("seed", -1))
    if seed == -1:
        # this one can safely use np.random because it produces a single value
        seed = np.random.randint(np.iinfo(np.int32).max)

    logger.info(
        "request from %s: %s steps of %s using %s in %s on %s, %sx%s, %s, %s - %s",
        user,
        steps,
        scheduler,
        model_path,
        pipeline,
        device or "any device",
        width,
        height,
        cfg,
        seed,
        prompt,
    )

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
        tiles=tiles,
        overlap=overlap,
        stride=stride,
    )
    size = Size(width, height)
    return (device, params, size)


def border_from_request() -> Border:
    left = get_and_clamp_int(
        request.args,
        "left",
        get_config_value("left"),
        get_config_value("left", "max"),
        get_config_value("left", "min"),
    )
    right = get_and_clamp_int(
        request.args,
        "right",
        get_config_value("right"),
        get_config_value("right", "max"),
        get_config_value("right", "min"),
    )
    top = get_and_clamp_int(
        request.args,
        "top",
        get_config_value("top"),
        get_config_value("top", "max"),
        get_config_value("top", "min"),
    )
    bottom = get_and_clamp_int(
        request.args,
        "bottom",
        get_config_value("bottom"),
        get_config_value("bottom", "max"),
        get_config_value("bottom", "min"),
    )

    return Border(left, right, top, bottom)


def upscale_from_request() -> UpscaleParams:
    denoise = get_and_clamp_float(
        request.args,
        "denoise",
        get_config_value("denoise"),
        get_config_value("denoise", "max"),
        get_config_value("denoise", "min"),
    )
    scale = get_and_clamp_int(
        request.args,
        "scale",
        get_config_value("scale"),
        get_config_value("scale", "max"),
        get_config_value("scale", "min"),
    )
    outscale = get_and_clamp_int(
        request.args,
        "outscale",
        get_config_value("outscale"),
        get_config_value("outscale", "max"),
        get_config_value("outscale", "min"),
    )
    upscaling = get_from_list(request.args, "upscaling", get_upscaling_models())
    correction = get_from_list(request.args, "correction", get_correction_models())
    faces = get_not_empty(request.args, "faces", "false") == "true"
    face_outscale = get_and_clamp_int(
        request.args,
        "faceOutscale",
        get_config_value("faceOutscale"),
        get_config_value("faceOutscale", "max"),
        get_config_value("faceOutscale", "min"),
    )
    face_strength = get_and_clamp_float(
        request.args,
        "faceStrength",
        get_config_value("faceStrength"),
        get_config_value("faceStrength", "max"),
        get_config_value("faceStrength", "min"),
    )
    upscale_order = request.args.get("upscaleOrder", "correction-first")

    return UpscaleParams(
        upscaling,
        correction_model=correction,
        denoise=denoise,
        faces=faces,
        face_outscale=face_outscale,
        face_strength=face_strength,
        format="onnx",
        outscale=outscale,
        scale=scale,
        upscale_order=upscale_order,
    )


def highres_from_request() -> HighresParams:
    iterations = get_and_clamp_int(
        request.args,
        "highresIterations",
        get_config_value("highresIterations"),
        get_config_value("highresIterations", "max"),
        get_config_value("highresIterations", "min"),
    )
    method = get_from_list(request.args, "highresMethod", get_highres_methods())
    scale = get_and_clamp_int(
        request.args,
        "highresScale",
        get_config_value("highresScale"),
        get_config_value("highresScale", "max"),
        get_config_value("highresScale", "min"),
    )
    steps = get_and_clamp_int(
        request.args,
        "highresSteps",
        get_config_value("highresSteps"),
        get_config_value("highresSteps", "max"),
        get_config_value("highresSteps", "min"),
    )
    strength = get_and_clamp_float(
        request.args,
        "highresStrength",
        get_config_value("highresStrength"),
        get_config_value("highresStrength", "max"),
        get_config_value("highresStrength", "min"),
    )
    return HighresParams(
        scale,
        steps,
        strength,
        method=method,
        iterations=iterations,
    )
