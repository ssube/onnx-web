from logging import getLogger
from typing import List, Optional

from PIL import Image

from ..image import valid_image
from ..params import ImageParams, StageParams
from ..server import ServerContext
from ..worker import ProgressCallback, WorkerContext

logger = getLogger(__name__)


def blend_linear(
    _job: WorkerContext,
    _server: ServerContext,
    _stage: StageParams,
    _params: ImageParams,
    *,
    alpha: float,
    sources: Optional[List[Image.Image]] = None,
    _callback: Optional[ProgressCallback] = None,
    **kwargs,
) -> Image.Image:
    logger.info("blending image using linear interpolation")

    resized = [valid_image(s) for s in sources]

    return Image.blend(resized[1], resized[0], alpha)
