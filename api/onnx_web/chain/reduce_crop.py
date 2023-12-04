from logging import getLogger
from typing import Optional

from PIL import Image

from ..params import ImageParams, Size, StageParams
from ..server import ServerContext
from ..worker import WorkerContext
from .base import BaseStage
from .result import StageResult

logger = getLogger(__name__)


class ReduceCropStage(BaseStage):
    def run(
        self,
        _worker: WorkerContext,
        _server: ServerContext,
        _stage: StageParams,
        _params: ImageParams,
        sources: StageResult,
        *,
        origin: Size,
        size: Size,
        stage_source: Optional[Image.Image] = None,
        **kwargs,
    ) -> StageResult:
        outputs = []

        for source in sources.as_image():
            image = source.crop((origin.width, origin.height, size.width, size.height))
            logger.info(
                "created thumbnail with dimensions: %sx%s", image.width, image.height
            )
            outputs.append(image)

        return StageResult(images=outputs)
