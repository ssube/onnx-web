from logging import getLogger

from PIL import Image

from ..params import ImageParams, StageParams, UpscaleParams
from ..worker import WorkerContext
from ..server import ServerContext

logger = getLogger(__name__)

device = "cpu"


def correct_codeformer(
    job: WorkerContext,
    _server: ServerContext,
    _stage: StageParams,
    _params: ImageParams,
    source: Image.Image,
    *,
    stage_source: Image.Image = None,
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
