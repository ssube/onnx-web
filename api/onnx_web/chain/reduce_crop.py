from logging import getLogger
from typing import List, Optional

from PIL import Image

from ..params import ImageParams, Size, StageParams
from ..server import ServerContext
from ..worker import WorkerContext
from .stage import BaseStage

logger = getLogger(__name__)


class ReduceCropStage(BaseStage):
    def run(
        self,
        _job: WorkerContext,
        _server: ServerContext,
        _stage: StageParams,
        _params: ImageParams,
        sources: List[Image.Image],
        *,
        origin: Size,
        size: Size,
        stage_source: Optional[Image.Image] = None,
        **kwargs,
    ) -> List[Image.Image]:
        outputs = []

        for source in sources:
            image = source.crop((origin.width, origin.height, size.width, size.height))
            logger.info(
                "created thumbnail with dimensions: %sx%s", image.width, image.height
            )
            outputs.append(image)

        return outputs
