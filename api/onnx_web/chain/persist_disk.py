from logging import getLogger
from typing import List

from PIL import Image

from ..output import save_image
from ..params import ImageParams, StageParams
from ..server import ServerContext
from ..worker import WorkerContext
from .stage import BaseStage

logger = getLogger(__name__)


class PersistDiskStage(BaseStage):
    def run(
        self,
        _worker: WorkerContext,
        server: ServerContext,
        _stage: StageParams,
        params: ImageParams,
        sources: List[Image.Image],
        *,
        outputs: List[str],
        stage_source: Image.Image,
        **kwargs,
    ) -> List[Image.Image]:
        for source, output in zip(sources, outputs):
            # TODO: append index to output name
            dest = save_image(server, output, source, params=params)
            logger.info("saved image to %s", dest)

        return sources
