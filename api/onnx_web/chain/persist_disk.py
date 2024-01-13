from logging import getLogger
from typing import Optional

from PIL import Image

from ..output import save_result
from ..params import ImageParams, SizeChart, StageParams
from ..server import ServerContext
from ..worker import WorkerContext
from .base import BaseStage
from .result import StageResult

logger = getLogger(__name__)


class PersistDiskStage(BaseStage):
    max_tile = SizeChart.max

    def run(
        self,
        worker: WorkerContext,
        server: ServerContext,
        _stage: StageParams,
        params: ImageParams,
        sources: StageResult,
        *,
        stage_source: Optional[Image.Image] = None,
        **kwargs,
    ) -> StageResult:
        logger.info("persisting %s images to disk", len(sources))

        save_result(server, sources, worker.job)

        return sources
