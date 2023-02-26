from os import path

from flask import Flask, send_from_directory

from ..worker.pool import DevicePoolExecutor
from .context import ServerContext
from .utils import wrap_route


def serve_bundle_file(context: ServerContext, filename="index.html"):
    return send_from_directory(path.join("..", context.bundle_path), filename)


# non-API routes
def index(context: ServerContext):
    return serve_bundle_file(context)


def index_path(context: ServerContext, filename: str):
    return serve_bundle_file(context, filename)


def output(context: ServerContext, filename: str):
    return send_from_directory(
        path.join("..", context.output_path), filename, as_attachment=False
    )


def register_static_routes(
    app: Flask, context: ServerContext, pool: DevicePoolExecutor
):
    return [
        app.route("/")(wrap_route(index, context)),
        app.route("/<path:filename>")(wrap_route(index_path, context)),
        app.route("/output/<path:filename>")(wrap_route(output, context)),
    ]
