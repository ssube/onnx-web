from logging import getLogger
from typing import Optional

import cv2
from PIL import Image

from ..params import ImageParams, SizeChart, StageParams
from ..server import ServerContext
from ..worker import ProgressCallback, WorkerContext
from .base import BaseStage
from .result import StageResult

logger = getLogger(__name__)


class BlendDenoiseFastNLMeansStage(BaseStage):
    max_tile = SizeChart.max

    def run(
        self,
        _worker: WorkerContext,
        _server: ServerContext,
        _stage: StageParams,
        _params: ImageParams,
        sources: StageResult,
        *,
        strength: int = 3,
        stage_source: Optional[Image.Image] = None,
        callback: Optional[ProgressCallback] = None,
        **kwargs,
    ) -> StageResult:
        logger.info("denoising source images")

        results = []
        for source in sources.as_arrays():
            data = cv2.cvtColor(source, cv2.COLOR_RGB2BGR)
            data = cv2.fastNlMeansDenoisingColored(data, None, strength, strength)
            results.append(cv2.cvtColor(data, cv2.COLOR_BGR2RGB))

        return StageResult.from_arrays(results, metadata=sources.metadata)
