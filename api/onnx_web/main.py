import atexit
import gc
from logging import getLogger

from diffusers.utils.logging import disable_progress_bar
from flask import Flask
from flask_cors import CORS
from huggingface_hub.utils.tqdm import disable_progress_bars
from setproctitle import setproctitle
from torch.multiprocessing import set_start_method

from .server.api import register_api_routes
from .server.context import ServerContext
from .server.hacks import apply_patches
from .server.load import (
    get_available_platforms,
    load_extras,
    load_models,
    load_params,
    load_platforms,
)
from .server.static import register_static_routes
from .server.utils import check_paths
from .utils import is_debug
from .worker import DevicePoolExecutor

logger = getLogger(__name__)


def main():
    setproctitle("onnx-web server")
    set_start_method("spawn", force=True)

    context = ServerContext.from_environ()
    apply_patches(context)
    check_paths(context)
    load_extras(context)
    load_models(context)
    load_params(context)
    load_platforms(context)

    if is_debug():
        gc.set_debug(gc.DEBUG_STATS)

    if not context.show_progress:
        disable_progress_bar()
        disable_progress_bars()

    app = Flask(__name__)
    CORS(app, origins=context.cors_origin)

    # any is a fake device, should not be in the pool
    pool = DevicePoolExecutor(
        context, [p for p in get_available_platforms() if p.device != "any"]
    )

    # register routes
    register_static_routes(app, context, pool)
    register_api_routes(app, context, pool)

    return app, pool


def run():
    app, pool = main()

    def quit():
        logger.info("shutting down workers")
        pool.join()

    atexit.register(quit)
    return app


if __name__ == "__main__":
    app, pool = main()
    app.run("0.0.0.0", 5000, debug=is_debug())
    logger.info("shutting down app")
    pool.join()
