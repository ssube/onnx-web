from typing import List, Optional

from PIL import Image

from .result import StageResult
from ..params import ImageParams, Size, SizeChart, StageParams
from ..server.context import ServerContext
from ..worker.context import WorkerContext


class BaseStage:
    max_tile = SizeChart.auto

    def run(
        self,
        _worker: WorkerContext,
        _server: ServerContext,
        _stage: StageParams,
        _params: ImageParams,
        _sources: List[Image.Image],
        *args,
        stage_source: Optional[Image.Image] = None,
        **kwargs,
    ) -> StageResult:
        raise NotImplementedError()  # noqa

    def steps(
        self,
        _params: ImageParams,
        _size: Size,
    ) -> int:
        return 1  # noqa

    def outputs(
        self,
        _params: ImageParams,
        sources: int,
    ) -> int:
        return sources
