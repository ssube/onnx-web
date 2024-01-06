from typing import Any

from ..params import HighresParams, ImageParams, Size, StageParams, UpscaleParams
from ..server import ServerContext
from ..worker import WorkerContext
from .result import StageResult


class EditMetadataStage:
    def run(
        self,
        _worker: WorkerContext,
        _server: ServerContext,
        _stage: StageParams,
        _params: ImageParams,
        source: StageResult,
        *,
        size: Size = None,
        upscale: UpscaleParams = None,
        highres: HighresParams = None,
        note: str = None,
        **kwargs,
    ) -> Any:
        # Modify the source image's metadata using the provided parameters
        for metadata in source.metadata:
            if note is not None:
                metadata.note = note

            if size is not None:
                metadata.size = size

            if upscale is not None:
                metadata.upscale = upscale

            if highres is not None:
                metadata.highres = highres

        # Return the modified source image
        return source
