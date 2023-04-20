from logging import getLogger

from flask import Flask, jsonify, make_response, request

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
    pool.join()
    pool.start()
    logger.info("restarted worker pool")


def worker_status(server: ServerContext, pool: DevicePoolExecutor):
    return jsonify(pool.status())


def register_admin_routes(app: Flask, server: ServerContext, pool: DevicePoolExecutor):
    return [
        app.route("/api/restart", methods=["POST"])(
            wrap_route(restart_workers, server, pool=pool)
        ),
        app.route("/api/status")(wrap_route(worker_status, server, pool=pool)),
    ]
