from typing import Optional

from ..params import (
    HighresParams,
    ImageParams,
    Size,
    SizeChart,
    StageParams,
    UpscaleParams,
)
from ..server import ServerContext
from ..worker import WorkerContext
from .base import BaseStage
from .result import StageResult


class EditMetadataStage(BaseStage):
    max_tile = SizeChart.max

    def run(
        self,
        _worker: WorkerContext,
        _server: ServerContext,
        _stage: StageParams,
        _params: ImageParams,
        source: StageResult,
        *,
        size: Optional[Size] = None,
        upscale: Optional[UpscaleParams] = None,
        highres: Optional[HighresParams] = None,
        note: Optional[str] = None,
        replace_params: Optional[ImageParams] = None,
        **kwargs,
    ) -> StageResult:
        # Modify the source image's metadata using the provided parameters
        for metadata in source.metadata:
            if note is not None:
                metadata.note = note

            if replace_params is not None:
                metadata.params = replace_params

            if size is not None:
                metadata.size = size

            if upscale is not None:
                metadata.upscale = upscale

            if highres is not None:
                metadata.highres = highres

        # Return the modified source image
        return source
