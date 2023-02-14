from logging import getLogger

from PIL import Image

from ..params import ImageParams, StageParams, UpscaleParams
from ..server.device_pool import JobContext
from ..utils import ServerContext

logger = getLogger(__name__)

device = "cpu"


def correct_codeformer(
    job: JobContext,
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

    device = job.get_device()
    pipe = CodeFormer(upscale=upscale.face_outscale).to(device.torch_device())
    return pipe(stage_source or source)
