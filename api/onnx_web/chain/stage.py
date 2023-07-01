from typing import Optional

from PIL import Image

from onnx_web.params import ImageParams, Size, SizeChart, StageParams
from onnx_web.server.context import ServerContext
from onnx_web.worker.context import WorkerContext


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
