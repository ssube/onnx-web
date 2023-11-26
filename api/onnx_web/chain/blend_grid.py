from logging import getLogger
from typing import Optional

from PIL import Image

from ..params import ImageParams, Size, SizeChart, StageParams
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

        images = sources.as_image()
        ref_image = images[0]
        size = Size(*ref_image.size)

        output = Image.new(ref_image.mode, (size.width * width, size.height * height))

        # TODO: labels
        if order is None:
            order = range(len(images))

        for i in range(len(order)):
            x = i % width
            y = i // width

            n = order[i]
            output.paste(images[n], (x * size.width, y * size.height))

        return StageResult(images=[*images, output])

    def outputs(
        self,
        _params: ImageParams,
        sources: int,
    ) -> int:
        return sources + 1
