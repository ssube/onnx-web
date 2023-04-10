from functools import partial, update_wrapper
from os import makedirs, path
from typing import Callable, Dict, List, Tuple

from flask import Flask

from ..utils import base_join
from ..worker.pool import DevicePoolExecutor
from .context import ServerContext


def check_paths(server: ServerContext) -> None:
    if not path.exists(server.model_path):
        raise RuntimeError("model path must exist")

    if not path.exists(server.output_path):
        makedirs(server.output_path)


def get_model_path(server: ServerContext, model: str):
    return base_join(server.model_path, model)


def register_routes(
    app: Flask,
    server: ServerContext,
    pool: DevicePoolExecutor,
    routes: List[Tuple[str, Dict, Callable]],
):
    for route, kwargs, method in routes:
        app.route(route, **kwargs)(wrap_route(method, server, pool=pool))


def wrap_route(func, *args, **kwargs):
    """
    From http://louistiao.me/posts/adding-__name__-and-__doc__-attributes-to-functoolspartial-objects/
    """
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func
