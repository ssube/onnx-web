from logging import getLogger
from typing import Optional

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
        stage_source: Optional[Image.Image] = None,
        stage_mask: Optional[Image.Image] = None,
        _callback: Optional[ProgressCallback] = None,
        **kwargs,
    ) -> StageResult:
        logger.info("blending image using mask")

        # TODO: does this need an alpha channel?
        mult_mask = Image.new(stage_mask.mode, stage_mask.size, color="black")
        mult_mask.alpha_composite(stage_mask)
        mult_mask = mult_mask.convert("L")

        if is_debug():
            save_image(server, "last-mask.png", stage_mask)
            save_image(server, "last-mult-mask.png", mult_mask)

        return StageResult(
            images=[
                Image.composite(stage_source, source, mult_mask)
                for source in sources.as_image()
            ]
        )
