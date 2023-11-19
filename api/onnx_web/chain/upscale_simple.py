from logging import getLogger
from typing import List, Optional

from PIL import Image

from ..params import ImageParams, StageParams, UpscaleParams
from ..server import ServerContext
from ..worker import WorkerContext
from .base import BaseStage
from .result import StageResult

logger = getLogger(__name__)


class UpscaleSimpleStage(BaseStage):
    def run(
        self,
        _worker: WorkerContext,
        _server: ServerContext,
        _stage: StageParams,
        _params: ImageParams,
        sources: StageResult,
        *,
        method: str,
        upscale: UpscaleParams,
        stage_source: Optional[Image.Image] = None,
        **kwargs,
    ) -> StageResult:
        if upscale.scale <= 1:
            logger.debug(
                "simple upscale stage run with scale of %s, skipping", upscale.scale
            )
            return sources

        outputs = []
        for source in sources.as_image():
            scaled_size = (source.width * upscale.scale, source.height * upscale.scale)

            if method == "bilinear":
                logger.debug("using bilinear interpolation for highres")
                source = source.resize(scaled_size, resample=Image.Resampling.BILINEAR)
            elif method == "lanczos":
                logger.debug("using Lanczos interpolation for highres")
                source = source.resize(scaled_size, resample=Image.Resampling.LANCZOS)
            else:
                logger.warning("unknown upscaling method: %s", method)

            outputs.append(source)

        return outputs
