from logging import getLogger
from typing import Optional

from PIL import Image

from ..params import ImageParams, StageParams
from ..server import ServerContext
from ..worker import ProgressCallback, WorkerContext
from .base import BaseStage
from .result import StageResult

logger = getLogger(__name__)


class BlendLinearStage(BaseStage):
    def run(
        self,
        _worker: WorkerContext,
        _server: ServerContext,
        _stage: StageParams,
        _params: ImageParams,
        sources: StageResult,
        *,
        alpha: float,
        stage_source: Optional[Image.Image] = None,
        _callback: Optional[ProgressCallback] = None,
        **kwargs,
    ) -> StageResult:
        logger.info("blending source images using linear interpolation")

        return StageResult(
            images=[
                Image.blend(source, stage_source, alpha)
                for source in sources.as_image()
            ]
        )
