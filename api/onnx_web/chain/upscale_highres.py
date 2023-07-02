from logging import getLogger
from typing import Optional

from PIL import Image

from ..chain.highres import stage_highres
from ..params import HighresParams, ImageParams, StageParams, UpscaleParams
from ..server import ServerContext
from ..worker import WorkerContext
from ..worker.context import ProgressCallback

logger = getLogger(__name__)


class UpscaleHighresStage:
    def run(
        self,
        job: WorkerContext,
        server: ServerContext,
        stage: StageParams,
        params: ImageParams,
        source: Image.Image,
        *,
        highres: HighresParams,
        upscale: UpscaleParams,
        stage_source: Optional[Image.Image] = None,
        callback: Optional[ProgressCallback] = None,
        **kwargs,
    ) -> Image.Image:
        source = stage_source or source

        if highres.scale <= 1:
            return source

        chain = stage_highres(stage, params, highres, upscale)

        return chain(
            job,
            server,
            params,
            source,
            callback=callback,
        )
