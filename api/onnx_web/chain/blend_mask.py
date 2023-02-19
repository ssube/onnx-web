from logging import getLogger
from typing import List, Optional

from PIL import Image

from onnx_web.image import valid_image
from onnx_web.output import save_image

from ..params import ImageParams, StageParams
from ..server import JobContext, ProgressCallback, ServerContext
from ..utils import is_debug

logger = getLogger(__name__)


def blend_mask(
    _job: JobContext,
    server: ServerContext,
    _stage: StageParams,
    _params: ImageParams,
    *,
    sources: Optional[List[Image.Image]] = None,
    mask: Optional[Image.Image] = None,
    _callback: ProgressCallback = None,
    **kwargs,
) -> Image.Image:
    logger.info("blending image using mask")

    mult_mask = Image.new("RGBA", mask.size, color="black")
    mult_mask.alpha_composite(mask)
    mult_mask = mult_mask.convert("L")

    if is_debug():
        save_image(server, "last-mask.png", mask)
        save_image(server, "last-mult-mask.png", mult_mask)

    resized = [
        valid_image(s, min_dims=mult_mask.size, max_dims=mult_mask.size)
        for s in sources
    ]

    return Image.composite(resized[0], resized[1], mult_mask)
