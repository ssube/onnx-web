from logging import getLogger

from PIL import Image

from ..params import ImageParams, Size, StageParams
from ..server import JobContext, ServerContext

logger = getLogger(__name__)


def reduce_crop(
    _job: JobContext,
    _server: ServerContext,
    _stage: StageParams,
    _params: ImageParams,
    source: Image.Image,
    *,
    origin: Size,
    size: Size,
    stage_source: Image.Image = None,
    **kwargs,
) -> Image.Image:
    source = stage_source or source

    image = source.crop((origin.width, origin.height, size.width, size.height))
    logger.info("created thumbnail with dimensions: %sx%s", image.width, image.height)
    return image
