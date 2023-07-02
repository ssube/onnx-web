from logging import getLogger
from typing import Optional

from PIL import Image

from ..params import ImageParams, StageParams, UpscaleParams
from ..server import ServerContext
from ..worker import WorkerContext
from .stage import BaseStage

logger = getLogger(__name__)


class UpscaleSimpleStage(BaseStage):
    def run(
        self,
        _job: WorkerContext,
        _server: ServerContext,
        _stage: StageParams,
        _params: ImageParams,
        source: Image.Image,
        *,
        method: str,
        upscale: UpscaleParams,
        stage_source: Optional[Image.Image] = None,
        **kwargs,
    ) -> Image.Image:
        source = stage_source or source

        if upscale.scale <= 1:
            logger.debug(
                "simple upscale stage run with scale of %s, skipping", upscale.scale
            )
            return source

        scaled_size = (source.width * upscale.scale, source.height * upscale.scale)

        if method == "bilinear":
            logger.debug("using bilinear interpolation for highres")
            source = source.resize(scaled_size, resample=Image.Resampling.BILINEAR)
        elif method == "lanczos":
            logger.debug("using Lanczos interpolation for highres")
            source = source.resize(scaled_size, resample=Image.Resampling.LANCZOS)
        else:
            logger.warning("unknown upscaling method: %s", method)

        return source
