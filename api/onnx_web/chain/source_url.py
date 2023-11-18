from io import BytesIO
from logging import getLogger
from typing import List, Optional

import requests
from PIL import Image

from ..params import ImageParams, StageParams
from ..server import ServerContext
from ..worker import WorkerContext
from .base import BaseStage

logger = getLogger(__name__)


class SourceURLStage(BaseStage):
    def run(
        self,
        _worker: WorkerContext,
        _server: ServerContext,
        _stage: StageParams,
        _params: ImageParams,
        sources: List[Image.Image],
        *,
        source_urls: List[str],
        stage_source: Optional[Image.Image] = None,
        **kwargs,
    ) -> List[Image.Image]:
        logger.info("loading image from URL source")

        if len(sources) > 0:
            logger.info(
                "source images were passed to a source stage, new images will be appended"
            )

        outputs = list(sources)
        for url in source_urls:
            response = requests.get(url)
            output = Image.open(BytesIO(response.content))

            logger.info("final output image size: %sx%s", output.width, output.height)
            outputs.append(output)

        return outputs

    def outputs(
        self,
        params: ImageParams,
        sources: int,
    ) -> int:
        return sources + 1
