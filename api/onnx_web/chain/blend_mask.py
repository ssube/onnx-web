from logging import getLogger
from typing import Optional, Tuple

from PIL import Image

from ..output import save_image
from ..params import ImageParams, StageParams
from ..server import ServerContext
from ..utils import is_debug
from ..worker import ProgressCallback, WorkerContext
from .base import BaseStage
from .result import StageResult

logger = getLogger(__name__)


class BlendMaskStage(BaseStage):
    def run(
        self,
        _worker: WorkerContext,
        server: ServerContext,
        _stage: StageParams,
        _params: ImageParams,
        sources: StageResult,
        *,
        dims: Tuple[int, int, int],
        stage_source: Optional[Image.Image] = None,
        stage_mask: Optional[Image.Image] = None,
        _callback: Optional[ProgressCallback] = None,
        **kwargs,
    ) -> StageResult:
        logger.info("blending image using mask")

        mult_mask = Image.new(stage_mask.mode, stage_mask.size, color="black")
        mult_mask = Image.alpha_composite(mult_mask, stage_mask)
        mult_mask = mult_mask.convert("L")

        top, left, tile = dims
        stage_source_tile = stage_source.crop((top, left, tile, tile))

        if is_debug():
            save_image(server, "last-mask.png", stage_mask)
            save_image(server, "last-mult-mask.png", mult_mask)
            save_image(server, "last-stage-source.png", stage_source_tile)

        return StageResult.from_images(
            [
                Image.composite(stage_source_tile, source, mult_mask)
                for source in sources.as_image()
            ]
        )
