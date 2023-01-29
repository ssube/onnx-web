from logging import getLogger
from PIL import Image


from ..params import (
    ImageParams,
    StageParams,
)
from ..utils import (
    base_join,
    ServerContext,
)

logger = getLogger(__name__)


def persist_disk(
    ctx: ServerContext,
    _stage: StageParams,
    _params: ImageParams,
    source_image: Image.Image,
    *,
    output: str,
    **kwargs,
) -> Image.Image:
    dest = base_join(ctx.output_path, output)
    source_image.save(dest)
    logger.info('saved image to %s', dest)
    return source_image
