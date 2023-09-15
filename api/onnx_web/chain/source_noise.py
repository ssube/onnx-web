from logging import getLogger
from typing import Callable, List, Optional

from PIL import Image

from ..params import ImageParams, Size, StageParams
from ..server import ServerContext
from ..worker import WorkerContext
from .stage import BaseStage

logger = getLogger(__name__)


class SourceNoiseStage(BaseStage):
    def run(
        self,
        _worker: WorkerContext,
        _server: ServerContext,
        _stage: StageParams,
        _params: ImageParams,
        sources: List[Image.Image],
        *,
        size: Size,
        noise_source: Callable,
        stage_source: Optional[Image.Image] = None,
        **kwargs,
    ) -> List[Image.Image]:
        logger.info("generating image from noise source")

        if len(sources) > 0:
            logger.info(
                "source images were passed to a source stage, new images will be appended"
            )

        outputs = list(sources)

        # TODO: looping over sources and ignoring params does not make much sense for a source stage
        for source in sources:
            output = noise_source(source, (size.width, size.height), (0, 0))

            logger.info("final output image size: %sx%s", output.width, output.height)
            outputs.append(output)

        return outputs

    def outputs(
        self,
        params: ImageParams,
        sources: int,
    ) -> int:
        return sources + 1
