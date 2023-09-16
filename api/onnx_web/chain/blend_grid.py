from logging import getLogger
from typing import List, Optional

from PIL import Image

from ..params import ImageParams, StageParams
from ..server import ServerContext
from ..worker import ProgressCallback, WorkerContext
from .stage import BaseStage

logger = getLogger(__name__)


class BlendGridStage(BaseStage):
    def run(
        self,
        _worker: WorkerContext,
        _server: ServerContext,
        _stage: StageParams,
        _params: ImageParams,
        sources: List[Image.Image],
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
    ) -> List[Image.Image]:
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

        return [*sources, output]

    def outputs(
        self,
        _params: ImageParams,
        sources: int,
    ) -> int:
        return sources + 1
