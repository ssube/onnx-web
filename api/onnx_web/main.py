import atexit
import gc
import mimetypes
from functools import partial
from logging import getLogger

from diffusers.utils.logging import disable_progress_bar
from flask import Flask
from flask_cors import CORS
from huggingface_hub.utils.tqdm import disable_progress_bars
from setproctitle import setproctitle
from torch.multiprocessing import set_start_method

from .server.admin import register_admin_routes
from .server.api import register_api_routes
from .server.context import ServerContext
from .server.hacks import apply_patches
from .server.load import (
    get_available_platforms,
    load_extras,
    load_models,
    load_params,
    load_platforms,
    load_wildcards,
)
from .server.plugin import load_plugins, register_plugins
from .server.static import register_static_routes
from .server.utils import check_paths
from .utils import is_debug
from .worker import DevicePoolExecutor

logger = getLogger(__name__)


def main():
    setproctitle("onnx-web server")
    set_start_method("spawn", force=True)

    # set up missing mimetypes
    mimetypes.add_type("application/javascript", ".js")
    mimetypes.add_type("text/css", ".css")

    # launch server, read env and list paths
    server = ServerContext.from_environ()
    apply_patches(server)
    check_paths(server)

    # debug options
    if server.debug:
        import debugpy
        debugpy.listen(5678)
        logger.warning("waiting for debugger")
        debugpy.wait_for_client()
        gc.set_debug(gc.DEBUG_STATS)

    # register plugins
    exports = load_plugins(server)
    success = register_plugins(exports)
    if success:
        logger.info("all plugins loaded successfully")
    else:
        logger.warning("error loading plugins")

    # load additional resources
    load_extras(server)
    load_models(server)
    load_params(server)
    load_platforms(server)
    load_wildcards(server)

    # misc server options
    if not server.show_progress:
        disable_progress_bar()
        disable_progress_bars()

    # create workers
    # any is a fake device and should not be in the pool
    pool = DevicePoolExecutor(
        server, [p for p in get_available_platforms() if p.device != "any"]
    )

    # create server
    app = Flask(__name__)
    CORS(app, origins=server.cors_origin)

    # register routes
    register_static_routes(app, server, pool)
    register_api_routes(app, server, pool)
    register_admin_routes(app, server, pool)

    return server, app, pool


def run():
    server, app, pool = main()
    pool.start()

    def quit(p: DevicePoolExecutor):
        logger.info("shutting down workers")
        p.join()

    logger.info(
        "starting %s API server with admin token: %s",
        server.server_version,
        server.admin_token,
    )
    atexit.register(partial(quit, pool))
    return app


if __name__ == "__main__":
    server, app, pool = main()
    logger.info("starting image workers")
    pool.start()
    logger.info(
        "starting %s API server with admin token: %s",
        server.server_version,
        server.admin_token,
    )
    app.run("0.0.0.0", 5000, debug=is_debug())
    logger.info("shutting down workers")
    pool.join()
