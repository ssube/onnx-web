from logging import getLogger

from flask import Flask, jsonify, make_response, request
from jsonschema import ValidationError, validate

from ..utils import load_config, load_config_str
from ..worker.pool import DevicePoolExecutor
from .context import ServerContext
from .utils import wrap_route

logger = getLogger(__name__)


def check_admin(server: ServerContext):
    return request.args.get("token", None) == server.admin_token


def restart_workers(server: ServerContext, pool: DevicePoolExecutor):
    if not check_admin(server):
        return make_response(jsonify({})), 401

    logger.info("restarting worker pool")
    pool.recycle(recycle_all=True)
    logger.info("restarted worker pool")

    return jsonify(pool.status())


def worker_status(server: ServerContext, pool: DevicePoolExecutor):
    if not check_admin(server):
        return make_response(jsonify({})), 401

    return jsonify(pool.status())


def get_extra_models(server: ServerContext):
    if not check_admin(server):
        return make_response(jsonify({})), 401

    with open(server.extra_models[0]) as f:
        resp = make_response(f.read())
        resp.content_type = "application/json"
        return resp


def update_extra_models(server: ServerContext):
    if not check_admin(server):
        return make_response(jsonify({})), 401

    extra_schema = load_config("./schemas/extras.yaml")

    try:
        data = load_config_str(request.json)
        try:
            validate(data, extra_schema)
        except ValidationError:
            logger.exception("invalid data in extras file")
    except Exception:
        logger.exception("TODO")

    # TODO: write to file
    return jsonify(server.extra_models)


def register_admin_routes(app: Flask, server: ServerContext, pool: DevicePoolExecutor):
    return [
        app.route("/api/extras")(wrap_route(get_extra_models, server)),
        app.route("/api/extras", methods=["PUT"])(
            wrap_route(update_extra_models, server)
        ),
        app.route("/api/restart", methods=["POST"])(
            wrap_route(restart_workers, server, pool=pool)
        ),
        app.route("/api/status")(wrap_route(worker_status, server, pool=pool)),
    ]
