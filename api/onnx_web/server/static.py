from os import path

from flask import Flask, send_from_directory

from ..worker.pool import DevicePoolExecutor
from .context import ServerContext
from .utils import wrap_route


def serve_bundle_file(server: ServerContext, filename="index.html"):
    return send_from_directory(path.join("..", server.bundle_path), filename)


# non-API routes
def index(server: ServerContext):
    return serve_bundle_file(server)


def index_path(server: ServerContext, filename: str):
    return serve_bundle_file(server, filename)


def output(server: ServerContext, filename: str):
    return send_from_directory(
        path.join("..", server.output_path), filename, as_attachment=False
    )


def register_static_routes(
    app: Flask, server: ServerContext, _pool: DevicePoolExecutor
):
    return [
        app.route("/")(wrap_route(index, server)),
        app.route("/<path:filename>")(wrap_route(index_path, server)),
        app.route("/output/<path:filename>")(wrap_route(output, server)),
    ]
