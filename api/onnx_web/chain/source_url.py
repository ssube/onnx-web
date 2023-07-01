from io import BytesIO
from logging import getLogger

import requests
from PIL import Image

from ..params import ImageParams, StageParams
from ..server import ServerContext
from ..worker import WorkerContext

logger = getLogger(__name__)


class SourceURLStage:
    def run(
        self,
        _job: WorkerContext,
        _server: ServerContext,
        _stage: StageParams,
        _params: ImageParams,
        source: Image.Image,
        *,
        source_url: str,
        stage_source: Image.Image,
        **kwargs,
    ) -> Image.Image:
        source = stage_source or source
        logger.info("loading image from URL source")

        if source is not None:
            logger.warn(
                "a source image was passed to a source stage, and will be discarded"
            )

        response = requests.get(source_url)
        output = Image.open(BytesIO(response.content))

        logger.info("final output image size: %sx%s", output.width, output.height)
        return output
