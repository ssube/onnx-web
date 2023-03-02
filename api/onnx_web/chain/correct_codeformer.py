from logging import getLogger
from typing import Optional

from PIL import Image

from ..params import ImageParams, StageParams, UpscaleParams
from ..server import ServerContext
from ..worker import WorkerContext

logger = getLogger(__name__)

device = "cpu"


def correct_codeformer(
    job: WorkerContext,
    _server: ServerContext,
    _stage: StageParams,
    _params: ImageParams,
    source: Image.Image,
    *,
    stage_source: Optional[Image.Image] = None,
    upscale: UpscaleParams,
    **kwargs,
) -> Image.Image:
    # must be within the load function for patch to take effect
    from codeformer import CodeFormer

    source = stage_source or source

    upscale = upscale.with_args(**kwargs)

    device = job.get_device()
    pipe = CodeFormer(upscale=upscale.face_outscale).to(device.torch_str())
    return pipe(source)
