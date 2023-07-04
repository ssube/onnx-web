from logging import getLogger
from typing import List

from PIL import Image

from ..params import ImageParams, Size, StageParams
from ..server import ServerContext
from ..worker import WorkerContext
from .stage import BaseStage

logger = getLogger(__name__)


class ReduceThumbnailStage(BaseStage):
    def run(
        self,
        _job: WorkerContext,
        _server: ServerContext,
        _stage: StageParams,
        _params: ImageParams,
        sources: List[Image.Image],
        *,
        size: Size,
        stage_source: Image.Image,
        **kwargs,
    ) -> List[Image.Image]:
        outputs = []

        for source in sources:
            image = source.copy()

            image = image.thumbnail((size.width, size.height))

            logger.info(
                "created thumbnail with dimensions: %sx%s", image.width, image.height
            )

            outputs.append(image)

        return outputs
