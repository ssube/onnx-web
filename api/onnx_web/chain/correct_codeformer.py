from logging import getLogger
from typing import List, Optional

from PIL import Image

from ..params import ImageParams, StageParams, UpscaleParams
from ..server import ServerContext
from ..worker import WorkerContext
from .base import BaseStage
from .result import StageResult

logger = getLogger(__name__)


class CorrectCodeformerStage(BaseStage):
    def run(
        self,
        worker: WorkerContext,
        _server: ServerContext,
        _stage: StageParams,
        _params: ImageParams,
        sources: StageResult,
        *,
        stage_source: Optional[Image.Image] = None,
        upscale: UpscaleParams,
        **kwargs,
    ) -> StageResult:
        # must be within the load function for patch to take effect
        # TODO: rewrite and remove
        from codeformer import CodeFormer

        upscale = upscale.with_args(**kwargs)

        device = worker.get_device()
        pipe = CodeFormer(upscale=upscale.face_outscale).to(device.torch_str())
        return StageResult(images=[pipe(source) for source in sources])
