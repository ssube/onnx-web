from logging import getLogger

from PIL import Image

from ..output import save_image
from ..params import ImageParams, StageParams
from ..server import ServerContext
from ..worker import WorkerContext

logger = getLogger(__name__)


def persist_disk(
    _job: WorkerContext,
    server: ServerContext,
    _stage: StageParams,
    _params: ImageParams,
    source: Image.Image,
    *,
    output: str,
    stage_source: Image.Image,
    **kwargs,
) -> Image.Image:
    source = stage_source or source

    dest = save_image(server, output, source)
    logger.info("saved image to %s", dest)
    return source
