from functools import partial, update_wrapper
from os import makedirs, path
from typing import Callable, Dict, List, Tuple

from flask import Flask

from ..utils import base_join
from ..worker.pool import DevicePoolExecutor
from .context import ServerContext


def check_paths(context: ServerContext) -> None:
    if not path.exists(context.model_path):
        raise RuntimeError("model path must exist")

    if not path.exists(context.output_path):
        makedirs(context.output_path)


def get_model_path(context: ServerContext, model: str):
    return base_join(context.model_path, model)


def register_routes(
    app: Flask,
    context: ServerContext,
    pool: DevicePoolExecutor,
    routes: List[Tuple[str, Dict, Callable]],
):
    for route, kwargs, method in routes:
        app.route(route, **kwargs)(wrap_route(method, context, pool=pool))


def wrap_route(func, *args, **kwargs):
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func
