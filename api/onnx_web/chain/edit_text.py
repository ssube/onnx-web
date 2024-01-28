from typing import Optional, Tuple

from PIL import ImageDraw

from ..params import ImageParams, SizeChart, StageParams
from ..server import ServerContext
from ..worker import ProgressCallback, WorkerContext
from .base import BaseStage
from .result import StageResult


class EditTextStage(BaseStage):
    max_tile = SizeChart.max

    def run(
        self,
        _worker: WorkerContext,
        _server: ServerContext,
        _stage: StageParams,
        _params: ImageParams,
        source: StageResult,
        *,
        text: str,
        position: Tuple[int, int],
        fill: str = "white",
        stroke: str = "black",
        stroke_width: int = 1,
        callback: Optional[ProgressCallback] = None,
        **kwargs,
    ) -> StageResult:
        # Add text to each image in source at the given position
        results = []

        for image in source.as_images():
            image = image.copy()
            draw = ImageDraw.Draw(image)
            draw.text(
                position, text, fill=fill, stroke_width=stroke_width, stroke_fill=stroke
            )
            results.append(image)

        return StageResult.from_images(results, source.metadata)
