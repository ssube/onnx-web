from typing import Optional

from PIL import Image

from ..params import ImageParams, Size, SizeChart, StageParams
from ..server.context import ServerContext
from ..worker.context import WorkerContext
from .result import StageResult


class BaseStage:
    max_tile = SizeChart.auto

    def run(
        self,
        _worker: WorkerContext,
        _server: ServerContext,
        _stage: StageParams,
        _params: ImageParams,
        _sources: StageResult,
        *,
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
