from logging import getLogger
from typing import List, Optional

from PIL import Image

from ..output import save_image
from ..params import ImageParams, Size, SizeChart, StageParams
from ..server import ServerContext
from ..worker import WorkerContext
from .base import BaseStage
from .result import StageResult

logger = getLogger(__name__)


class PersistDiskStage(BaseStage):
    max_tile = SizeChart.max

    def run(
        self,
        _worker: WorkerContext,
        server: ServerContext,
        _stage: StageParams,
        params: ImageParams,
        sources: StageResult,
        *,
        output: List[str],
        size: Optional[Size] = None,
        stage_source: Optional[Image.Image] = None,
        **kwargs,
    ) -> StageResult:
        logger.info(
            "persisting %s images to disk: %s", len(sources), output
        )

        for source, name in zip(sources, output):
            dest = save_image(server, name, source, params=params, size=size)
            logger.info("saved image to %s", dest)

        return sources
