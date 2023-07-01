from logging import getLogger
from typing import List, Optional

from PIL import Image

from ..params import ImageParams, StageParams
from ..server import ServerContext
from ..worker import ProgressCallback, WorkerContext

logger = getLogger(__name__)


class BlendLinearStage:
    def run(
        self,
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

        return Image.blend(sources[1], sources[0], alpha)
