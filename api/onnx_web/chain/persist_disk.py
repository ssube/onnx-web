from logging import getLogger
from typing import List, Optional

from PIL import Image

from ..output import save_image
from ..params import ImageParams, Size, SizeChart, StageParams
from ..server import ServerContext
from ..worker import WorkerContext
from .stage import BaseStage

logger = getLogger(__name__)


class PersistDiskStage(BaseStage):
    max_tile = SizeChart.max

    def run(
        self,
        _worker: WorkerContext,
        server: ServerContext,
        _stage: StageParams,
        params: ImageParams,
        sources: List[Image.Image],
        *,
        output: List[str],
        size: Optional[Size] = None,
        stage_source: Optional[Image.Image] = None,
        **kwargs,
    ) -> List[Image.Image]:
        logger.info(
            "persisting images to disk: %s, %s", [s.size for s in sources], output
        )

        for source, name in zip(sources, output):
            dest = save_image(server, name, source, params=params, size=size)
            logger.info("saved image to %s", dest)

        return sources
