from logging import getLogger
from typing import Tuple

import numpy as np
from flask import request

from ..diffusers.load import pipeline_schedulers
from ..params import Border, DeviceParams, ImageParams, Size, UpscaleParams
from ..utils import get_and_clamp_float, get_and_clamp_int, get_from_list, get_not_empty
from .context import ServerContext
from .load import (
    get_available_platforms,
    get_config_value,
    get_correction_models,
    get_upscaling_models,
)
from .utils import get_model_path

logger = getLogger(__name__)


def pipeline_from_request(
    context: ServerContext,
) -> Tuple[DeviceParams, ImageParams, Size]:
    user = request.remote_addr

    # platform stuff
    device = None
    device_name = request.args.get("platform")

    if device_name is not None and device_name != "any":
        for platform in get_available_platforms():
            if platform.device == device_name:
                device = platform

    # pipeline stuff
    lpw = get_not_empty(request.args, "lpw", "false") == "true"
    model = get_not_empty(request.args, "model", get_config_value("model"))
    model_path = get_model_path(context, model)
    scheduler = get_from_list(
        request.args, "scheduler", list(pipeline_schedulers.keys())
    )

    if scheduler is None:
        scheduler = get_config_value("scheduler")

    inversion = request.args.get("inversion", None)
    inversion_path = None
    if inversion is not None and inversion.strip() != "":
        inversion_path = get_model_path(context, inversion)

    # image params
    prompt = get_not_empty(request.args, "prompt", get_config_value("prompt"))
    negative_prompt = request.args.get("negativePrompt", None)

    if negative_prompt is not None and negative_prompt.strip() == "":
        negative_prompt = None

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

    seed = int(request.args.get("seed", -1))
    if seed == -1:
        # this one can safely use np.random because it produces a single value
        seed = np.random.randint(np.iinfo(np.int32).max)

    logger.info(
        "request from %s: %s rounds of %s using %s on %s, %sx%s, %s, %s - %s",
        user,
        steps,
        scheduler,
        model_path,
        device or "any device",
        width,
        height,
        cfg,
        seed,
        prompt,
    )

    params = ImageParams(
        model_path,
        scheduler,
        prompt,
        cfg,
        steps,
        seed,
        eta=eta,
        lpw=lpw,
        negative_prompt=negative_prompt,
        batch=batch,
        inversion=inversion_path,
    )
    size = Size(width, height)
    return (device, params, size)


def border_from_request() -> Border:
    left = get_and_clamp_int(
        request.args, "left", 0, get_config_value("width", "max"), 0
    )
    right = get_and_clamp_int(
        request.args, "right", 0, get_config_value("width", "max"), 0
    )
    top = get_and_clamp_int(
        request.args, "top", 0, get_config_value("height", "max"), 0
    )
    bottom = get_and_clamp_int(
        request.args, "bottom", 0, get_config_value("height", "max"), 0
    )

    return Border(left, right, top, bottom)


def upscale_from_request() -> UpscaleParams:
    denoise = get_and_clamp_float(request.args, "denoise", 0.5, 1.0, 0.0)
    scale = get_and_clamp_int(request.args, "scale", 1, 4, 1)
    outscale = get_and_clamp_int(request.args, "outscale", 1, 4, 1)
    upscaling = get_from_list(request.args, "upscaling", get_upscaling_models())
    correction = get_from_list(request.args, "correction", get_correction_models())
    faces = get_not_empty(request.args, "faces", "false") == "true"
    face_outscale = get_and_clamp_int(request.args, "faceOutscale", 1, 4, 1)
    face_strength = get_and_clamp_float(request.args, "faceStrength", 0.5, 1.0, 0.0)
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
