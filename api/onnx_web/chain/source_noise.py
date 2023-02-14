from logging import getLogger
from typing import Callable

from PIL import Image

from ..params import ImageParams, Size, StageParams
from ..server.device_pool import JobContext
from ..utils import ServerContext

logger = getLogger(__name__)


def source_noise(
    _job: JobContext,
    _server: ServerContext,
    _stage: StageParams,
    params: ImageParams,
    source_image: Image.Image,
    *,
    size: Size,
    noise_source: Callable,
    **kwargs,
) -> Image.Image:
    logger.info("generating image from noise source")

    if source_image is not None:
        logger.warn("a source image was passed to a noise stage, but will be discarded")

    output = noise_source(source_image, (size.width, size.height), (0, 0))

    logger.info("final output image size: %sx%s", output.width, output.height)
    return output
