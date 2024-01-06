from logging import getLogger

from PIL import Image

from ..params import ImageParams, Size, StageParams
from ..server import ServerContext
from ..worker import WorkerContext
from .base import BaseStage
from .result import StageResult

logger = getLogger(__name__)


class ReduceThumbnailStage(BaseStage):
    def run(
        self,
        _worker: WorkerContext,
        _server: ServerContext,
        _stage: StageParams,
        _params: ImageParams,
        sources: StageResult,
        *,
        size: Size,
        stage_source: Image.Image,
        **kwargs,
    ) -> StageResult:
        outputs = []

        for source in sources.as_images():
            image = source.copy()

            image = image.thumbnail((size.width, size.height))

            logger.info(
                "created thumbnail with dimensions: %sx%s", image.width, image.height
            )

            outputs.append(image)

        return StageResult.from_images(outputs, metadata=sources.metadata)
