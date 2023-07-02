from typing import Optional

from PIL import Image

from ..params import ImageParams, Size, SizeChart, StageParams
from ..server.context import ServerContext
from ..worker.context import WorkerContext


class BaseStage:
    max_tile = SizeChart.auto

    def run(
        self,
        job: WorkerContext,
        server: ServerContext,
        stage: StageParams,
        _params: ImageParams,
        source: Image.Image,
        *args,
        stage_source: Optional[Image.Image] = None,
        **kwargs,
    ) -> Image.Image:
        raise NotImplementedError()

    def steps(
        self,
        _params: ImageParams,
        size: Size,
    ) -> int:
        raise NotImplementedError()
