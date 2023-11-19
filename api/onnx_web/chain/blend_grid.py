from logging import getLogger
from typing import Optional

from PIL import Image

from ..params import ImageParams, SizeChart, StageParams
from ..server import ServerContext
from ..worker import ProgressCallback, WorkerContext
from .base import BaseStage
from .result import StageResult

logger = getLogger(__name__)


class BlendGridStage(BaseStage):
    max_tile = SizeChart.max

    def run(
        self,
        _worker: WorkerContext,
        _server: ServerContext,
        _stage: StageParams,
        _params: ImageParams,
        sources: StageResult,
        *,
        height: int,
        width: int,
        # rows: Optional[List[str]] = None,
        # columns: Optional[List[str]] = None,
        # title: Optional[str] = None,
        order: Optional[int] = None,
        stage_source: Optional[Image.Image] = None,
        callback: Optional[ProgressCallback] = None,
        **kwargs,
    ) -> StageResult:
        logger.info("combining source images using grid layout")

        size = sources[0].size

        output = Image.new("RGB", (size[0] * width, size[1] * height))

        # TODO: labels
        if order is None:
            order = range(len(sources))

        for i in range(len(order)):
            x = i % width
            y = i // width

            n = order[i]
            output.paste(sources[n], (x * size[0], y * size[1]))

        return StageResult(images=[*sources, output])

    def outputs(
        self,
        _params: ImageParams,
        sources: int,
    ) -> int:
        return sources + 1
