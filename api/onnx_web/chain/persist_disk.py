from logging import getLogger

from PIL import Image

from ..output import save_image
from ..params import ImageParams, StageParams
from ..server.device_pool import JobContext
from ..utils import ServerContext

logger = getLogger(__name__)


def persist_disk(
    _job: JobContext,
    server: ServerContext,
    _stage: StageParams,
    _params: ImageParams,
    source_image: Image.Image,
    *,
    output: str,
    **kwargs,
) -> Image.Image:
    dest = save_image(server, output, source_image)
    logger.info("saved image to %s", dest)
    return source_image
