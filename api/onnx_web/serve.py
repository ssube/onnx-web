import gc
from functools import cmp_to_key
from glob import glob
from io import BytesIO
from logging import getLogger
from os import makedirs, path
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
import yaml
from flask import Flask, jsonify, make_response, request, send_from_directory, url_for
from flask_cors import CORS
from jsonschema import validate
from onnxruntime import get_available_providers
from PIL import Image

from .chain import (
    ChainPipeline,
    blend_img2img,
    blend_inpaint,
    correct_codeformer,
    correct_gfpgan,
    persist_disk,
    persist_s3,
    reduce_crop,
    reduce_thumbnail,
    source_noise,
    source_txt2img,
    upscale_outpaint,
    upscale_resrgan,
    upscale_stable_diffusion,
)
from .diffusion.load import pipeline_schedulers
from .diffusion.run import (
    run_blend_pipeline,
    run_img2img_pipeline,
    run_inpaint_pipeline,
    run_txt2img_pipeline,
    run_upscale_pipeline,
)
from .image import (  # mask filters; noise sources
    mask_filter_gaussian_multiply,
    mask_filter_gaussian_screen,
    mask_filter_none,
    noise_source_fill_edge,
    noise_source_fill_mask,
    noise_source_gaussian,
    noise_source_histogram,
    noise_source_normal,
    noise_source_uniform,
)
from .output import json_params, make_output_name
from .params import (
    Border,
    DeviceParams,
    ImageParams,
    Size,
    StageParams,
    TileOrder,
    UpscaleParams,
)
from .server.device_pool import DevicePoolExecutor
from .server.hacks import apply_patches
from .utils import (
    ServerContext,
    base_join,
    get_and_clamp_float,
    get_and_clamp_int,
    get_from_list,
    get_from_map,
    get_not_empty,
    get_size,
    is_debug,
)

logger = getLogger(__name__)

# config caching
config_params: Dict[str, Dict[str, Union[float, int, str]]] = {}

# pipeline params
platform_providers = {
    "cpu": "CPUExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "directml": "DmlExecutionProvider",
    "rocm": "ROCMExecutionProvider",
}

noise_sources = {
    "fill-edge": noise_source_fill_edge,
    "fill-mask": noise_source_fill_mask,
    "gaussian": noise_source_gaussian,
    "histogram": noise_source_histogram,
    "normal": noise_source_normal,
    "uniform": noise_source_uniform,
}
mask_filters = {
    "none": mask_filter_none,
    "gaussian-multiply": mask_filter_gaussian_multiply,
    "gaussian-screen": mask_filter_gaussian_screen,
}
chain_stages = {
    "blend-img2img": blend_img2img,
    "blend-inpaint": blend_inpaint,
    "correct-codeformer": correct_codeformer,
    "correct-gfpgan": correct_gfpgan,
    "persist-disk": persist_disk,
    "persist-s3": persist_s3,
    "reduce-crop": reduce_crop,
    "reduce-thumbnail": reduce_thumbnail,
    "source-noise": source_noise,
    "source-txt2img": source_txt2img,
    "upscale-outpaint": upscale_outpaint,
    "upscale-resrgan": upscale_resrgan,
    "upscale-stable-diffusion": upscale_stable_diffusion,
}

# Available ORT providers
available_platforms: List[DeviceParams] = []

# loaded from model_path
diffusion_models: List[str] = []
correction_models: List[str] = []
upscaling_models: List[str] = []


def get_config_value(key: str, subkey: str = "default", default=None):
    return config_params.get(key, {}).get(subkey, default)


def url_from_rule(rule) -> str:
    options = {}
    for arg in rule.arguments:
        options[arg] = ":%s" % (arg)

    return url_for(rule.endpoint, **options)


def pipeline_from_request() -> Tuple[DeviceParams, ImageParams, Size]:
    user = request.remote_addr

    # platform stuff
    device = None
    device_name = request.args.get("platform")

    if device_name is not None and device_name != "any":
        for platform in available_platforms:
            if platform.device == device_name:
                device = platform

    # pipeline stuff
    lpw = get_not_empty(request.args, "lpw", "false") == "true"
    model = get_not_empty(request.args, "model", get_config_value("model"))
    model_path = get_model_path(model)
    scheduler = get_from_map(
        request.args, "scheduler", pipeline_schedulers, get_config_value("scheduler")
    )

    # image params
    prompt = get_not_empty(request.args, "prompt", get_config_value("prompt"))
    negative_prompt = request.args.get("negativePrompt", None)

    if negative_prompt is not None and negative_prompt.strip() == "":
        negative_prompt = None

    cfg = get_and_clamp_float(
        request.args,
        "cfg",
        get_config_value("cfg"),
        get_config_value("cfg", "max"),
        get_config_value("cfg", "min"),
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
        scheduler.__name__,
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
        lpw=lpw,
        negative_prompt=negative_prompt,
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
    upscaling = get_from_list(request.args, "upscaling", upscaling_models)
    correction = get_from_list(request.args, "correction", correction_models)
    faces = get_not_empty(request.args, "faces", "false") == "true"
    face_outscale = get_and_clamp_int(request.args, "faceOutscale", 1, 4, 1)
    face_strength = get_and_clamp_float(request.args, "faceStrength", 0.5, 1.0, 0.0)

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
    )


def check_paths(context: ServerContext):
    if not path.exists(context.model_path):
        raise RuntimeError("model path must exist")

    if not path.exists(context.output_path):
        makedirs(context.output_path)


def get_model_name(model: str) -> str:
    base = path.basename(model)
    (file, _ext) = path.splitext(base)
    return file


def load_models(context: ServerContext):
    global diffusion_models
    global correction_models
    global upscaling_models

    diffusion_models = [
        get_model_name(f) for f in glob(path.join(context.model_path, "diffusion-*"))
    ]
    diffusion_models.extend(
        [
            get_model_name(f)
            for f in glob(path.join(context.model_path, "stable-diffusion-*"))
        ]
    )
    diffusion_models = list(set(diffusion_models))
    diffusion_models.sort()

    correction_models = [
        get_model_name(f) for f in glob(path.join(context.model_path, "correction-*"))
    ]
    correction_models = list(set(correction_models))
    correction_models.sort()

    upscaling_models = [
        get_model_name(f) for f in glob(path.join(context.model_path, "upscaling-*"))
    ]
    upscaling_models = list(set(upscaling_models))
    upscaling_models.sort()


def load_params(context: ServerContext):
    global config_params
    params_file = path.join(context.params_path, "params.json")
    with open(params_file, "r") as f:
        config_params = yaml.safe_load(f)

        if "platform" in config_params and context.default_platform is not None:
            logger.info(
                "Overriding default platform from environment: %s",
                context.default_platform,
            )
            config_platform = config_params.get("platform", {})
            config_platform["default"] = context.default_platform


def load_platforms(context: ServerContext):
    global available_platforms

    providers = list(get_available_providers())

    for potential in platform_providers:
        if (
            platform_providers[potential] in providers
            and potential not in context.block_platforms
        ):
            if potential == "cuda":
                for i in range(torch.cuda.device_count()):
                    available_platforms.append(
                        DeviceParams(
                            potential,
                            platform_providers[potential],
                            {
                                "device_id": i,
                            },
                        )
                    )
            else:
                available_platforms.append(
                    DeviceParams(potential, platform_providers[potential])
                )

    if context.any_platform:
        # the platform should be ignored when the job is scheduled, but set to CPU just in case
        available_platforms.append(DeviceParams("any", platform_providers["cpu"]))

    # make sure CPU is last on the list
    def any_first_cpu_last(a: DeviceParams, b: DeviceParams):
        if a.device == b.device:
            return 0

        # any should be first, if it's available
        if a.device == "any":
            return -1

        # cpu should be last, if it's available
        if a.device == "cpu":
            return 1

        return -1

    available_platforms = sorted(
        available_platforms, key=cmp_to_key(any_first_cpu_last)
    )

    logger.info(
        "available acceleration platforms: %s",
        ", ".join([str(p) for p in available_platforms]),
    )


context = ServerContext.from_environ()
apply_patches(context)
check_paths(context)
load_models(context)
load_params(context)
load_platforms(context)

app = Flask(__name__)
CORS(app, origins=context.cors_origin)

# any is a fake device, should not be in the pool
executor = DevicePoolExecutor([p for p in available_platforms if p.device != "any"])

if is_debug():
    gc.set_debug(gc.DEBUG_STATS)


def ready_reply(ready: bool, progress: int = 0):
    return jsonify(
        {
            "progress": progress,
            "ready": ready,
        }
    )


def error_reply(err: str):
    response = make_response(
        jsonify(
            {
                "error": err,
            }
        )
    )
    response.status_code = 400
    return response


def get_model_path(model: str):
    return base_join(context.model_path, model)


def serve_bundle_file(filename="index.html"):
    return send_from_directory(path.join("..", context.bundle_path), filename)


# routes


@app.route("/")
def index():
    return serve_bundle_file()


@app.route("/<path:filename>")
def index_path(filename):
    return serve_bundle_file(filename)


@app.route("/api")
def introspect():
    return {
        "name": "onnx-web",
        "routes": [
            {"path": url_from_rule(rule), "methods": list(rule.methods).sort()}
            for rule in app.url_map.iter_rules()
        ],
    }


@app.route("/api/settings/masks")
def list_mask_filters():
    return jsonify(list(mask_filters.keys()))


@app.route("/api/settings/models")
def list_models():
    return jsonify(
        {
            "diffusion": diffusion_models,
            "correction": correction_models,
            "upscaling": upscaling_models,
        }
    )


@app.route("/api/settings/noises")
def list_noise_sources():
    return jsonify(list(noise_sources.keys()))


@app.route("/api/settings/params")
def list_params():
    return jsonify(config_params)


@app.route("/api/settings/platforms")
def list_platforms():
    return jsonify([p.device for p in available_platforms])


@app.route("/api/settings/schedulers")
def list_schedulers():
    return jsonify(list(pipeline_schedulers.keys()))


@app.route("/api/img2img", methods=["POST"])
def img2img():
    if "source" not in request.files:
        return error_reply("source image is required")

    source_file = request.files.get("source")
    source_image = Image.open(BytesIO(source_file.read())).convert("RGB")

    device, params, size = pipeline_from_request()
    upscale = upscale_from_request()

    strength = get_and_clamp_float(
        request.args,
        "strength",
        get_config_value("strength"),
        get_config_value("strength", "max"),
        get_config_value("strength", "min"),
    )

    output = make_output_name(context, "img2img", params, size, extras=(strength,))
    logger.info("img2img job queued for: %s", output)

    source_image.thumbnail((size.width, size.height))
    executor.submit(
        output,
        run_img2img_pipeline,
        context,
        params,
        output,
        upscale,
        source_image,
        strength,
        needs_device=device,
    )

    return jsonify(json_params(output, params, size, upscale=upscale))


@app.route("/api/txt2img", methods=["POST"])
def txt2img():
    device, params, size = pipeline_from_request()
    upscale = upscale_from_request()

    output = make_output_name(context, "txt2img", params, size)
    logger.info("txt2img job queued for: %s", output)

    executor.submit(
        output,
        run_txt2img_pipeline,
        context,
        params,
        size,
        output,
        upscale,
        needs_device=device,
    )

    return jsonify(json_params(output, params, size, upscale=upscale))


@app.route("/api/inpaint", methods=["POST"])
def inpaint():
    if "source" not in request.files:
        return error_reply("source image is required")

    if "mask" not in request.files:
        return error_reply("mask image is required")

    source_file = request.files.get("source")
    source_image = Image.open(BytesIO(source_file.read())).convert("RGB")

    mask_file = request.files.get("mask")
    mask_image = Image.open(BytesIO(mask_file.read())).convert("RGB")

    device, params, size = pipeline_from_request()
    expand = border_from_request()
    upscale = upscale_from_request()

    fill_color = get_not_empty(request.args, "fillColor", "white")
    mask_filter = get_from_map(request.args, "filter", mask_filters, "none")
    noise_source = get_from_map(request.args, "noise", noise_sources, "histogram")
    strength = get_and_clamp_float(
        request.args,
        "strength",
        get_config_value("strength"),
        get_config_value("strength", "max"),
        get_config_value("strength", "min"),
    )
    tile_order = get_from_list(
        request.args, "tileOrder", [TileOrder.grid, TileOrder.kernel, TileOrder.spiral]
    )

    output = make_output_name(
        context,
        "inpaint",
        params,
        size,
        extras=(
            expand.left,
            expand.right,
            expand.top,
            expand.bottom,
            mask_filter.__name__,
            noise_source.__name__,
            strength,
            fill_color,
            tile_order,
        ),
    )
    logger.info("inpaint job queued for: %s", output)

    source_image.thumbnail((size.width, size.height))
    mask_image.thumbnail((size.width, size.height))
    executor.submit(
        output,
        run_inpaint_pipeline,
        context,
        params,
        size,
        output,
        upscale,
        source_image,
        mask_image,
        expand,
        noise_source,
        mask_filter,
        strength,
        fill_color,
        tile_order,
        needs_device=device,
    )

    return jsonify(json_params(output, params, size, upscale=upscale, border=expand))


@app.route("/api/upscale", methods=["POST"])
def upscale():
    if "source" not in request.files:
        return error_reply("source image is required")

    source_file = request.files.get("source")
    source_image = Image.open(BytesIO(source_file.read())).convert("RGB")

    device, params, size = pipeline_from_request()
    upscale = upscale_from_request()

    output = make_output_name(context, "upscale", params, size)
    logger.info("upscale job queued for: %s", output)

    source_image.thumbnail((size.width, size.height))
    executor.submit(
        output,
        run_upscale_pipeline,
        context,
        params,
        size,
        output,
        upscale,
        source_image,
        needs_device=device,
    )

    return jsonify(json_params(output, params, size, upscale=upscale))


@app.route("/api/chain", methods=["POST"])
def chain():
    logger.debug(
        "chain pipeline request: %s, %s", request.form.keys(), request.files.keys()
    )
    body = request.form.get("chain") or request.files.get("chain")
    if body is None:
        return error_reply("chain pipeline must have a body")

    data = yaml.safe_load(body)
    with open("./schemas/chain.yaml", "r") as f:
        schema = yaml.safe_load(f.read())

    logger.debug("validating chain request: %s against %s", data, schema)
    validate(data, schema)

    # get defaults from the regular parameters
    device, params, size = pipeline_from_request()
    output = make_output_name(context, "chain", params, size)

    pipeline = ChainPipeline()
    for stage_data in data.get("stages", []):
        callback = chain_stages[stage_data.get("type")]
        kwargs = stage_data.get("params", {})
        logger.info("request stage: %s, %s", callback.__name__, kwargs)

        stage = StageParams(
            stage_data.get("name", callback.__name__),
            tile_size=get_size(kwargs.get("tile_size")),
            outscale=get_and_clamp_int(kwargs, "outscale", 1, 4),
        )

        if "border" in kwargs:
            border = Border.even(int(kwargs.get("border")))
            kwargs["border"] = border

        if "upscale" in kwargs:
            upscale = UpscaleParams(kwargs.get("upscale"))
            kwargs["upscale"] = upscale

        stage_source_name = "source:%s" % (stage.name)
        stage_mask_name = "mask:%s" % (stage.name)

        if stage_source_name in request.files:
            logger.debug(
                "loading source image %s for pipeline stage %s",
                stage_source_name,
                stage.name,
            )
            source_file = request.files.get(stage_source_name)
            source_image = Image.open(BytesIO(source_file.read())).convert("RGB")
            source_image.thumbnail((size.width, size.height))
            kwargs["source_image"] = source_image

        if stage_mask_name in request.files:
            logger.debug(
                "loading mask image %s for pipeline stage %s",
                stage_mask_name,
                stage.name,
            )
            mask_file = request.files.get(stage_mask_name)
            mask_image = Image.open(BytesIO(mask_file.read())).convert("RGB")
            mask_image.thumbnail((size.width, size.height))
            kwargs["mask_image"] = mask_image

        pipeline.append((callback, stage, kwargs))

    logger.info("running chain pipeline with %s stages", len(pipeline.stages))

    # build and run chain pipeline
    empty_source = Image.new("RGB", (size.width, size.height))
    executor.submit(
        output,
        pipeline,
        context,
        params,
        empty_source,
        output=output,
        size=size,
        needs_device=device,
    )

    return jsonify(json_params(output, params, size))


@app.route("/api/blend", methods=["POST"])
def blend():
    if "mask" not in request.files:
        return error_reply("mask image is required")

    mask_file = request.files.get("mask")
    mask = Image.open(BytesIO(mask_file.read())).convert("RGBA")

    max_sources = 2
    sources = []

    for i in range(max_sources):
        source_file = request.files.get("source:%s" % (i))
        sources.append(Image.open(BytesIO(source_file.read())).convert("RGBA"))

    device, params, size = pipeline_from_request()
    upscale = upscale_from_request()

    output = make_output_name(context, "upscale", params, size)
    logger.info("upscale job queued for: %s", output)

    executor.submit(
        output,
        run_blend_pipeline,
        context,
        params,
        size,
        output,
        upscale,
        sources,
        mask,
        needs_device=device,
    )

    return jsonify(json_params(output, params, size, upscale=upscale))


@app.route("/api/cancel", methods=["PUT"])
def cancel():
    output_file = request.args.get("output", None)

    cancel = executor.cancel(output_file)

    return ready_reply(cancel)


@app.route("/api/ready")
def ready():
    output_file = request.args.get("output", None)

    done, progress = executor.done(output_file)

    if done is None:
        file = base_join(context.output_path, output_file)
        if path.exists(file):
            return ready_reply(True)

    return ready_reply(done, progress=progress)


@app.route("/api/status")
def status():
    return jsonify(executor.status())


@app.route("/output/<path:filename>")
def output(filename: str):
    return send_from_directory(
        path.join("..", context.output_path), filename, as_attachment=False
    )
