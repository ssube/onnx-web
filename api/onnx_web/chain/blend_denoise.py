from logging import getLogger
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image

from ..params import ImageParams, SizeChart, StageParams
from ..server import ServerContext
from ..worker import ProgressCallback, WorkerContext
from .stage import BaseStage

logger = getLogger(__name__)


class BlendDenoiseStage(BaseStage):
    max_tile = SizeChart.max

    def run(
        self,
        _worker: WorkerContext,
        _server: ServerContext,
        _stage: StageParams,
        _params: ImageParams,
        sources: List[Image.Image],
        *,
        strength: int = 3,
        stage_source: Optional[Image.Image] = None,
        callback: Optional[ProgressCallback] = None,
        **kwargs,
    ) -> List[Image.Image]:
        logger.info("denoising source images")

        results = []
        for source in sources:
            data = cv2.cvtColor(np.array(source), cv2.COLOR_RGB2BGR)
            data = cv2.fastNlMeansDenoisingColored(data, None, strength, strength)
            results.append(Image.fromarray(cv2.cvtColor(data, cv2.COLOR_BGR2RGB)))

        return results
