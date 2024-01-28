from io import BytesIO
from logging import getLogger
from typing import List, Optional

import requests
from PIL import Image

from ..params import ImageParams, StageParams
from ..server import ServerContext
from ..worker import ProgressCallback, WorkerContext
from .base import BaseStage
from .result import ImageMetadata, StageResult

logger = getLogger(__name__)


class SourceURLStage(BaseStage):
    def run(
        self,
        _worker: WorkerContext,
        _server: ServerContext,
        _stage: StageParams,
        _params: ImageParams,
        sources: StageResult,
        *,
        source_urls: List[str],
        stage_source: Optional[Image.Image] = None,
        callback: Optional[ProgressCallback] = None,
        **kwargs,
    ) -> StageResult:
        logger.info("loading image from URL source")

        if len(sources) > 0:
            logger.info(
                "source images were passed to a source stage, new images will be appended"
            )

        outputs = sources.as_images()
        for url in source_urls:
            response = requests.get(url)
            output = Image.open(BytesIO(response.content))

            logger.info("final output image size: %sx%s", output.width, output.height)
            outputs.append(output)

        metadata = [ImageMetadata.unknown_image()] * len(outputs)
        return StageResult(images=outputs, metadata=metadata)

    def outputs(
        self,
        params: ImageParams,
        sources: int,
    ) -> int:
        return sources + 1
