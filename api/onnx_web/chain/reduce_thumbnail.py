from logging import getLogger

from PIL import Image

from ..params import ImageParams, Size, StageParams
from ..server.device_pool import JobContext
from ..utils import ServerContext

logger = getLogger(__name__)


def reduce_thumbnail(
    _job: JobContext,
    _server: ServerContext,
    _stage: StageParams,
    _params: ImageParams,
    source_image: Image.Image,
    *,
    size: Size,
    **kwargs,
) -> Image.Image:
    image = source_image.copy()
    image = image.thumbnail((size.width, size.height))
    logger.info("created thumbnail with dimensions: %sx%s", image.width, image.height)
    return image
