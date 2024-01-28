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
        tile_mask: Optional[Image.Image] = None,
        callback: Optional[ProgressCallback] = None,
        **kwargs,
    ) -> StageResult:
        logger.info("blending image using mask")

        mask_source = tile_mask or stage_mask
        mult_mask = Image.new(mask_source.mode, mask_source.size, color="black")
        mult_mask = Image.alpha_composite(mult_mask, mask_source)
        mult_mask = mult_mask.convert("L")

        left, top, tile = dims
        stage_source_tile = stage_source.crop((left, top, left + tile, top + tile))

        if is_debug():
            save_image(server, "last-mask.png", mask_source)
            save_image(server, "last-mult-mask.png", mult_mask)
            save_image(server, "last-stage-source.png", stage_source_tile)

        return StageResult.from_images(
            [
                Image.composite(stage_source_tile, source, mult_mask)
                for source in sources.as_images()
            ],
            metadata=sources.metadata,
        )
