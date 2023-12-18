from io import BytesIO
from logging import getLogger
from os import path
from typing import Any, Dict

from flask import Flask, jsonify, make_response, request, url_for
from jsonschema import validate
from PIL import Image

from ..chain import CHAIN_STAGES, ChainPipeline
from ..chain.result import StageResult
from ..diffusers.load import get_available_pipelines, get_pipeline_schedulers
from ..diffusers.run import (
    run_blend_pipeline,
    run_img2img_pipeline,
    run_inpaint_pipeline,
    run_txt2img_pipeline,
    run_upscale_pipeline,
)
from ..diffusers.utils import replace_wildcards
from ..output import json_params, make_output_name
from ..params import Size, StageParams, TileOrder
from ..transformers.run import run_txt2txt_pipeline
from ..utils import (
    base_join,
    get_and_clamp_float,
    get_and_clamp_int,
    get_boolean,
    get_from_list,
    get_from_map,
    get_not_empty,
    get_size,
    load_config,
    load_config_str,
    sanitize_name,
)
from ..worker.pool import DevicePoolExecutor
from .context import ServerContext
from .load import (
    get_available_platforms,
    get_config_params,
    get_config_value,
    get_correction_models,
    get_diffusion_models,
    get_extra_strings,
    get_mask_filters,
    get_network_models,
    get_noise_sources,
    get_source_filters,
    get_upscaling_models,
    get_wildcard_data,
)
from .params import (
    build_border,
    build_highres,
    build_upscale,
    pipeline_from_json,
    pipeline_from_request,
)
from .utils import wrap_route

logger = getLogger(__name__)


def ready_reply(
    ready: bool = False,
    cancelled: bool = False,
    failed: bool = False,
    pending: bool = False,
    progress: int = 0,
):
    return jsonify(
        {
            "cancelled": cancelled,
            "failed": failed,
            "pending": pending,
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


def url_from_rule(rule) -> str:
    options = {}
    for arg in rule.arguments:
        options[arg] = ":%s" % (arg)

    return url_for(rule.endpoint, **options)


def introspect(server: ServerContext, app: Flask):
    return {
        "name": "onnx-web",
        "routes": [
            {"path": url_from_rule(rule), "methods": list(rule.methods or []).sort()}
            for rule in app.url_map.iter_rules()
        ],
    }


def list_extra_strings(server: ServerContext):
    return jsonify(get_extra_strings())


def list_filters(server: ServerContext):
    mask_filters = list(get_mask_filters().keys())
    source_filters = list(get_source_filters().keys())
    return jsonify(
        {
            "mask": mask_filters,
            "source": source_filters,
        }
    )


def list_mask_filters(server: ServerContext):
    logger.info("dedicated list endpoint for mask filters is deprecated")
    return jsonify(list(get_mask_filters().keys()))


def list_models(server: ServerContext):
    return jsonify(
        {
            "correction": get_correction_models(),
            "diffusion": get_diffusion_models(),
            "networks": [model.tojson() for model in get_network_models()],
            "upscaling": get_upscaling_models(),
        }
    )


def list_noise_sources(server: ServerContext):
    return jsonify(list(get_noise_sources().keys()))


def list_params(server: ServerContext):
    return jsonify(get_config_params())


def list_pipelines(server: ServerContext):
    return jsonify(get_available_pipelines())


def list_platforms(server: ServerContext):
    return jsonify([p.device for p in get_available_platforms()])


def list_schedulers(server: ServerContext):
    return jsonify(get_pipeline_schedulers())


def list_wildcards(server: ServerContext):
    return jsonify(list(get_wildcard_data().keys()))


def img2img(server: ServerContext, pool: DevicePoolExecutor):
    source_file = request.files.get("source")
    if source_file is None:
        return error_reply("source image is required")

    source = Image.open(BytesIO(source_file.read())).convert("RGB")
    size = Size(source.width, source.height)

    device, params, _size = pipeline_from_request(server, "img2img")
    upscale = build_upscale()
    highres = build_highres()
    source_filter = get_from_list(
        request.args, "sourceFilter", list(get_source_filters().keys())
    )

    strength = get_and_clamp_float(
        request.args,
        "strength",
        get_config_value("strength"),
        get_config_value("strength", "max"),
        get_config_value("strength", "min"),
    )

    replace_wildcards(params, get_wildcard_data())

    output_count = params.batch
    if source_filter is not None and source_filter != "none":
        logger.debug(
            "including filtered source with outputs, filter: %s", source_filter
        )
        output_count += 1

    output = make_output_name(
        server, "img2img", params, size, extras=[strength], count=output_count
    )

    job_name = output[0]
    pool.submit(
        job_name,
        run_img2img_pipeline,
        server,
        params,
        output,
        upscale,
        highres,
        source,
        strength,
        needs_device=device,
        source_filter=source_filter,
    )

    logger.info("img2img job queued for: %s", job_name)

    return jsonify(json_params(output, params, size, upscale=upscale, highres=highres))


def txt2img(server: ServerContext, pool: DevicePoolExecutor):
    device, params, size = pipeline_from_request(server, "txt2img")
    upscale = build_upscale()
    highres = build_highres()

    replace_wildcards(params, get_wildcard_data())

    output = make_output_name(server, "txt2img", params, size, count=params.batch)

    job_name = output[0]
    pool.submit(
        job_name,
        run_txt2img_pipeline,
        server,
        params,
        size,
        output,
        upscale,
        highres,
        needs_device=device,
    )

    logger.info("txt2img job queued for: %s", job_name)

    return jsonify(json_params(output, params, size, upscale=upscale, highres=highres))


def inpaint(server: ServerContext, pool: DevicePoolExecutor):
    source_file = request.files.get("source")
    if source_file is None:
        return error_reply("source image is required")

    mask_file = request.files.get("mask")
    if mask_file is None:
        return error_reply("mask image is required")

    source = Image.open(BytesIO(source_file.read())).convert("RGBA")
    size = Size(source.width, source.height)

    mask_top_layer = Image.open(BytesIO(mask_file.read())).convert("RGBA")
    mask = Image.new("RGBA", mask_top_layer.size, color=(0, 0, 0, 255))
    mask.alpha_composite(mask_top_layer)
    mask.convert(mode="L")

    full_res_inpaint = get_boolean(
        request.args, "fullresInpaint", get_config_value("fullresInpaint")
    )
    full_res_inpaint_padding = get_and_clamp_float(
        request.args,
        "fullresInpaintPadding",
        get_config_value("fullresInpaintPadding"),
        get_config_value("fullresInpaintPadding", "max"),
        get_config_value("fullresInpaintPadding", "min"),
    )

    device, params, _size = pipeline_from_request(server, "inpaint")
    expand = build_border()
    upscale = build_upscale()
    highres = build_highres()

    fill_color = get_not_empty(request.args, "fillColor", "white")
    mask_filter = get_from_map(request.args, "filter", get_mask_filters(), "none")
    noise_source = get_from_map(request.args, "noise", get_noise_sources(), "histogram")
    tile_order = get_from_list(
        request.args, "tileOrder", [TileOrder.grid, TileOrder.kernel, TileOrder.spiral]
    )
    tile_order = TileOrder.spiral

    replace_wildcards(params, get_wildcard_data())

    output = make_output_name(
        server,
        "inpaint",
        params,
        size,
        extras=[
            expand.left,
            expand.right,
            expand.top,
            expand.bottom,
            mask_filter.__name__,
            noise_source.__name__,
            fill_color,
            tile_order,
        ],
    )

    job_name = output[0]
    pool.submit(
        job_name,
        run_inpaint_pipeline,
        server,
        params,
        size,
        output,
        upscale,
        highres,
        source,
        mask,
        expand,
        noise_source,
        mask_filter,
        fill_color,
        tile_order,
        full_res_inpaint,
        full_res_inpaint_padding,
        needs_device=device,
    )

    logger.info("inpaint job queued for: %s", job_name)

    return jsonify(
        json_params(
            output, params, size, upscale=upscale, border=expand, highres=highres
        )
    )


def upscale(server: ServerContext, pool: DevicePoolExecutor):
    source_file = request.files.get("source")
    if source_file is None:
        return error_reply("source image is required")

    source = Image.open(BytesIO(source_file.read())).convert("RGB")

    device, params, size = pipeline_from_request(server)
    upscale = build_upscale()
    highres = build_highres()

    replace_wildcards(params, get_wildcard_data())

    output = make_output_name(server, "upscale", params, size)

    job_name = output[0]
    pool.submit(
        job_name,
        run_upscale_pipeline,
        server,
        params,
        size,
        output,
        upscale,
        highres,
        source,
        needs_device=device,
    )

    logger.info("upscale job queued for: %s", job_name)

    return jsonify(json_params(output, params, size, upscale=upscale, highres=highres))


# keys that are specially parsed by params and should not show up in with_args
CHAIN_POP_KEYS = ["model", "control"]


def chain(server: ServerContext, pool: DevicePoolExecutor):
    if request.is_json:
        logger.debug("chain pipeline request with JSON body")
        data = request.get_json()
    else:
        logger.debug(
            "chain pipeline request: %s, %s", request.form.keys(), request.files.keys()
        )

        body = request.form.get("chain") or request.files.get("chain")
        if body is None:
            return error_reply("chain pipeline must have a body")

        data = load_config_str(body)

    schema = load_config("./schemas/chain.yaml")
    logger.debug("validating chain request: %s against %s", data, schema)
    validate(data, schema)

    device, base_params, base_size = pipeline_from_json(
        server, data=data.get("defaults")
    )

    # start building the pipeline
    pipeline = ChainPipeline()
    for stage_data in data.get("stages", []):
        stage_class = CHAIN_STAGES[stage_data.get("type")]
        kwargs: Dict[str, Any] = stage_data.get("params", {})
        logger.info("request stage: %s, %s", stage_class.__name__, kwargs)

        # TODO: combine base params with stage params
        _device, params, size = pipeline_from_json(server, data=kwargs)
        replace_wildcards(params, get_wildcard_data())

        # remove parsed keys, like model names (which become paths)
        for pop_key in CHAIN_POP_KEYS:
            if pop_key in kwargs:
                kwargs.pop(pop_key)

        if "seed" in kwargs and kwargs["seed"] == -1:
            kwargs.pop("seed")

        # replace kwargs with parsed versions
        kwargs["params"] = params
        kwargs["size"] = size

        border = build_border(kwargs)
        kwargs["border"] = border

        upscale = build_upscale(kwargs)
        kwargs["upscale"] = upscale

        # prepare the stage metadata
        stage = StageParams(
            stage_data.get("name", stage_class.__name__),
            tile_size=get_size(kwargs.get("tiles")),
            outscale=get_and_clamp_int(kwargs, "outscale", 1, 4),
        )

        # load any images related to this stage
        stage_source_name = "source:%s" % (stage.name)
        stage_mask_name = "mask:%s" % (stage.name)

        if stage_source_name in request.files:
            logger.debug(
                "loading source image %s for pipeline stage %s",
                stage_source_name,
                stage.name,
            )
            source_file = request.files.get(stage_source_name)
            if source_file is not None:
                source = Image.open(BytesIO(source_file.read())).convert("RGB")
                kwargs["stage_source"] = source

        if stage_mask_name in request.files:
            logger.debug(
                "loading mask image %s for pipeline stage %s",
                stage_mask_name,
                stage.name,
            )
            mask_file = request.files.get(stage_mask_name)
            if mask_file is not None:
                mask = Image.open(BytesIO(mask_file.read())).convert("RGB")
                kwargs["stage_mask"] = mask

        pipeline.append((stage_class(), stage, kwargs))

    logger.info("running chain pipeline with %s stages", len(pipeline.stages))

    output = make_output_name(
        server, "chain", base_params, base_size, count=pipeline.outputs(base_params, 0)
    )
    job_name = output[0]

    # build and run chain pipeline
    pool.submit(
        job_name,
        pipeline,
        server,
        base_params,
        StageResult.empty(),
        output=output,
        size=base_size,
        needs_device=device,
    )

    step_params = base_params.with_args(steps=pipeline.steps(base_params, base_size))
    return jsonify(json_params(output, step_params, base_size))


def blend(server: ServerContext, pool: DevicePoolExecutor):
    mask_file = request.files.get("mask")
    if mask_file is None:
        return error_reply("mask image is required")

    mask = Image.open(BytesIO(mask_file.read())).convert("RGBA")

    max_sources = 2
    sources = []

    for i in range(max_sources):
        source_file = request.files.get("source:%s" % (i))
        if source_file is None:
            logger.warning("missing source %s", i)
        else:
            source = Image.open(BytesIO(source_file.read())).convert("RGB")
            sources.append(source)

    device, params, size = pipeline_from_request(server)
    upscale = build_upscale()

    output = make_output_name(server, "upscale", params, size)
    job_name = output[0]
    pool.submit(
        job_name,
        run_blend_pipeline,
        server,
        params,
        size,
        output,
        upscale,
        # TODO: highres
        sources,
        mask,
        needs_device=device,
    )

    logger.info("upscale job queued for: %s", job_name)

    return jsonify(json_params(output, params, size, upscale=upscale))


def txt2txt(server: ServerContext, pool: DevicePoolExecutor):
    device, params, size = pipeline_from_request(server)

    output = make_output_name(server, "txt2txt", params, size)
    job_name = output[0]
    logger.info("upscale job queued for: %s", job_name)

    pool.submit(
        job_name,
        run_txt2txt_pipeline,
        server,
        params,
        size,
        output,
        needs_device=device,
    )

    return jsonify(json_params(output, params, size))


def cancel(server: ServerContext, pool: DevicePoolExecutor):
    output_file = request.args.get("output", None)
    if output_file is None:
        return error_reply("output name is required")

    output_file = sanitize_name(output_file)
    cancelled = pool.cancel(output_file)

    return ready_reply(cancelled=cancelled)


def ready(server: ServerContext, pool: DevicePoolExecutor):
    output_file = request.args.get("output", None)
    if output_file is None:
        return error_reply("output name is required")

    output_file = sanitize_name(output_file)
    pending, progress = pool.done(output_file)

    if pending:
        return ready_reply(pending=True)

    if progress is None:
        output = base_join(server.output_path, output_file)
        if path.exists(output):
            return ready_reply(ready=True)
        else:
            return ready_reply(
                ready=True,
                failed=True,
            )  # is a missing image really an error? yes will display the retry button

    return ready_reply(
        ready=progress.finished,
        progress=progress.progress,
        failed=progress.failed,
        cancelled=progress.cancelled,
    )


def register_api_routes(app: Flask, server: ServerContext, pool: DevicePoolExecutor):
    return [
        app.route("/api")(wrap_route(introspect, server, app=app)),
        app.route("/api/settings/filters")(wrap_route(list_filters, server)),
        app.route("/api/settings/masks")(wrap_route(list_mask_filters, server)),
        app.route("/api/settings/models")(wrap_route(list_models, server)),
        app.route("/api/settings/noises")(wrap_route(list_noise_sources, server)),
        app.route("/api/settings/params")(wrap_route(list_params, server)),
        app.route("/api/settings/pipelines")(wrap_route(list_pipelines, server)),
        app.route("/api/settings/platforms")(wrap_route(list_platforms, server)),
        app.route("/api/settings/schedulers")(wrap_route(list_schedulers, server)),
        app.route("/api/settings/strings")(wrap_route(list_extra_strings, server)),
        app.route("/api/settings/wildcards")(wrap_route(list_wildcards, server)),
        app.route("/api/img2img", methods=["POST"])(
            wrap_route(img2img, server, pool=pool)
        ),
        app.route("/api/txt2img", methods=["POST"])(
            wrap_route(txt2img, server, pool=pool)
        ),
        app.route("/api/txt2txt", methods=["POST"])(
            wrap_route(txt2txt, server, pool=pool)
        ),
        app.route("/api/inpaint", methods=["POST"])(
            wrap_route(inpaint, server, pool=pool)
        ),
        app.route("/api/upscale", methods=["POST"])(
            wrap_route(upscale, server, pool=pool)
        ),
        app.route("/api/chain", methods=["POST"])(wrap_route(chain, server, pool=pool)),
        app.route("/api/blend", methods=["POST"])(wrap_route(blend, server, pool=pool)),
        app.route("/api/cancel", methods=["PUT"])(
            wrap_route(cancel, server, pool=pool)
        ),
        app.route("/api/ready")(wrap_route(ready, server, pool=pool)),
    ]
