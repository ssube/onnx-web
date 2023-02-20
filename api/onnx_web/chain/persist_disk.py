from logging import getLogger
from typing import List

from PIL import Image

from ..output import save_image
from ..params import ImageParams, StageParams
from ..server import JobContext, ServerContext

logger = getLogger(__name__)


def persist_disk(
    _job: JobContext,
    server: ServerContext,
    _stage: StageParams,
    _params: ImageParams,
    source: Image.Image,
    *,
    output: List[str],
    stage_source: Image.Image,
    **kwargs,
) -> Image.Image:
    source = stage_source or source

    dest = save_image(server, output[0], source)
    logger.info("saved image to %s", dest)
    return source
