from logging import getLogger
from typing import Callable

from PIL import Image

from ..params import ImageParams, Size, StageParams
from ..server import ServerContext
from ..worker import WorkerContext

logger = getLogger(__name__)


def source_noise(
    _job: WorkerContext,
    _server: ServerContext,
    _stage: StageParams,
    _params: ImageParams,
    source: Image.Image,
    *,
    size: Size,
    noise_source: Callable,
    stage_source: Image.Image,
    **kwargs,
) -> Image.Image:
    source = stage_source or source
    logger.info("generating image from noise source")

    if source is not None:
        logger.warn("a source image was passed to a noise stage, but will be discarded")

    output = noise_source(source, (size.width, size.height), (0, 0))

    logger.info("final output image size: %sx%s", output.width, output.height)
    return output
