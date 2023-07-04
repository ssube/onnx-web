from logging import getLogger
from typing import List, Optional

from PIL import Image

from ..params import ImageParams, StageParams
from ..server import ServerContext
from ..worker import ProgressCallback, WorkerContext
from .stage import BaseStage

logger = getLogger(__name__)


class BlendLinearStage(BaseStage):
    def run(
        self,
        _job: WorkerContext,
        _server: ServerContext,
        _stage: StageParams,
        _params: ImageParams,
        sources: List[Image.Image],
        *,
        alpha: float,
        stage_source: Optional[Image.Image] = None,
        _callback: Optional[ProgressCallback] = None,
        **kwargs,
    ) -> List[Image.Image]:
        logger.info("blending source images using linear interpolation")

        return [Image.blend(source, stage_source, alpha) for source in sources]
