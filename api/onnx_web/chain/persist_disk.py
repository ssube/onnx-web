from logging import getLogger
from PIL import Image

from ..device_pool import (
    JobContext,
)
from ..params import (
    ImageParams,
    StageParams,
)
from ..output import (
    save_image,
)
from ..utils import (
    ServerContext,
)

logger = getLogger(__name__)


def persist_disk(
    _job: JobContext,
    ctx: ServerContext,
    _stage: StageParams,
    _params: ImageParams,
    source_image: Image.Image,
    *,
    output: str,
    **kwargs,
) -> Image.Image:
    dest = save_image(ctx, output, source_image)
    logger.info('saved image to %s', dest)
    return source_image
