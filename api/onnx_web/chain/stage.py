from typing import List, Optional

from PIL import Image

from ..params import ImageParams, Size, SizeChart, StageParams
from ..server.context import ServerContext
from ..worker.context import WorkerContext


class BaseStage:
    max_tile = SizeChart.auto

    def run(
        self,
        worker: WorkerContext,
        server: ServerContext,
        stage: StageParams,
        _params: ImageParams,
        sources: List[Image.Image],
        *args,
        stage_source: Optional[Image.Image] = None,
        **kwargs,
    ) -> List[Image.Image]:
        raise NotImplementedError()  # noqa

    def steps(
        self,
        params: ImageParams,
        size: Size,
    ) -> int:
        return 1  # noqa

    def outputs(
        self,
        params: ImageParams,
        sources: int,
    ) -> int:
        return sources
