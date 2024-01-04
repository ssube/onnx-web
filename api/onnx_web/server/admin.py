from logging import getLogger

from flask import Flask, jsonify, make_response, request
from jsonschema import ValidationError, validate

from ..utils import load_config, load_config_str
from ..worker.pool import DevicePoolExecutor
from .context import ServerContext
from .load import load_extras, load_models, load_wildcards
from .utils import wrap_route

logger = getLogger(__name__)

conversion_lock = False


def check_admin(server: ServerContext):
    return request.args.get("token", None) == server.admin_token


def restart_workers(server: ServerContext, pool: DevicePoolExecutor):
    if not check_admin(server):
        return make_response(jsonify({})), 401

    logger.info("restarting worker pool")
    pool.recycle(recycle_all=True)
    logger.info("restarted worker pool")

    return jsonify(pool.summary())


def worker_status(server: ServerContext, pool: DevicePoolExecutor):
    if not check_admin(server):
        return make_response(jsonify({})), 401

    return jsonify(pool.summary())


def get_extra_models(server: ServerContext):
    if not check_admin(server):
        return make_response(jsonify({})), 401

    with open(server.extra_models[0]) as f:
        resp = make_response(f.read())
        resp.content_type = "application/json"
        return resp


def update_extra_models(server: ServerContext):
    global conversion_lock

    if not check_admin(server):
        return make_response(jsonify({})), 401

    if conversion_lock:
        return make_response(jsonify({})), 409

    extra_schema = load_config("./schemas/extras.yaml")
    data_str = request.data.decode(encoding=(request.content_encoding or "utf-8"))

    try:
        data = load_config_str(data_str)
        try:
            validate(data, extra_schema)
        except ValidationError:
            logger.exception("invalid data in extras file")
    except Exception:
        logger.exception("error validating extras file")

    # TODO: make a backup
    with open(server.extra_models[0], mode="w") as f:
        f.write(data_str)

    logger.warning("downloading and converting models to ONNX")
    conversion_lock = True

    from onnx_web.convert.__main__ import main as convert

    convert(
        args=[
            "--correction",
            "--diffusion",
            "--upscaling",
            "--extras",
            *server.extra_models,
        ]
    )

    logger.info("finished converting models, reloading server")
    load_models(server)
    load_wildcards(server)
    load_extras(server)

    conversion_lock = False

    return jsonify(data)


def register_admin_routes(app: Flask, server: ServerContext, pool: DevicePoolExecutor):
    return [
        app.route("/api/extras")(wrap_route(get_extra_models, server)),
        app.route("/api/extras", methods=["PUT"])(
            wrap_route(update_extra_models, server)
        ),
        app.route("/api/worker/restart", methods=["POST"])(
            wrap_route(restart_workers, server, pool=pool)
        ),
        app.route("/api/worker/status")(wrap_route(worker_status, server, pool=pool)),
    ]
