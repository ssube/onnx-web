import gc

from diffusers.utils.logging import disable_progress_bar
from flask import Flask
from flask_cors import CORS
from huggingface_hub.utils.tqdm import disable_progress_bars

from .server.api import register_api_routes
from .server.static import register_static_routes
from .server.config import get_available_platforms, load_models, load_params, load_platforms
from .server.utils import check_paths
from .server.context import ServerContext
from .server.hacks import apply_patches
from .utils import (
    is_debug,
)
from .worker import DevicePoolExecutor


def main():
  context = ServerContext.from_environ()
  apply_patches(context)
  check_paths(context)
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
  pool = DevicePoolExecutor([p for p in get_available_platforms() if p.device != "any"])

  # register routes
  register_static_routes(app, context, pool)
  register_api_routes(app, context, pool)

  return app #, context, pool


if __name__ == "__main__":
  # app, context, pool = main()
  app = main()
  app.run("0.0.0.0", 5000, debug=is_debug())
  # pool.join()

