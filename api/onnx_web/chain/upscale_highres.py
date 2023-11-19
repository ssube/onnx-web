from logging import getLogger
from typing import Optional

from PIL import Image

from ..chain.highres import stage_highres
from ..params import HighresParams, ImageParams, StageParams, UpscaleParams
from ..server import ServerContext
from ..worker import WorkerContext
from ..worker.context import ProgressCallback
from .base import BaseStage
from .result import StageResult

logger = getLogger(__name__)


class UpscaleHighresStage(BaseStage):
    def run(
        self,
        worker: WorkerContext,
        server: ServerContext,
        stage: StageParams,
        params: ImageParams,
        sources: StageResult,
        *args,
        highres: HighresParams,
        upscale: UpscaleParams,
        stage_source: Optional[Image.Image] = None,
        callback: Optional[ProgressCallback] = None,
        **kwargs,
    ) -> StageResult:
        if highres.scale <= 1:
            return sources

        chain = stage_highres(stage, params, highres, upscale)

        outputs = [
            chain(
                worker,
                server,
                params,
                source,
                callback=callback,
            )
            for source in sources
        ]

        return StageResult(images=outputs)
