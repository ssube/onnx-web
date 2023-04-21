from logging import getLogger
from typing import List, Optional

from PIL import Image

from ..image import valid_image
from ..output import save_image
from ..params import ImageParams, StageParams
from ..server import ServerContext
from ..utils import is_debug
from ..worker import ProgressCallback, WorkerContext

logger = getLogger(__name__)


def blend_mask(
    _job: WorkerContext,
    server: ServerContext,
    _stage: StageParams,
    _params: ImageParams,
    *,
    sources: Optional[List[Image.Image]] = None,
    stage_mask: Optional[Image.Image] = None,
    _callback: Optional[ProgressCallback] = None,
    **kwargs,
) -> Image.Image:
    logger.info("blending image using mask")

    mult_mask = Image.new("RGBA", stage_mask.size, color="black")
    mult_mask.alpha_composite(stage_mask)
    mult_mask = mult_mask.convert("L")

    if is_debug():
        save_image(server, "last-mask.png", stage_mask)
        save_image(server, "last-mult-mask.png", mult_mask)

    resized = [
        valid_image(s, min_dims=mult_mask.size, max_dims=mult_mask.size)
        for s in sources
    ]

    return Image.composite(resized[1], resized[0], mult_mask)
