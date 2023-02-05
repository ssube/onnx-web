from logging import getLogger

from codeformer import CodeFormer
from PIL import Image

from ..device_pool import JobContext
from ..params import ImageParams, StageParams
from ..utils import ServerContext

logger = getLogger(__name__)

device = "cpu"


def correct_codeformer(
    job: JobContext,
    _server: ServerContext,
    stage: StageParams,
    _params: ImageParams,
    source: Image.Image,
    *,
    source_image: Image.Image = None,
    **kwargs,
) -> Image.Image:
    device = job.get_device()
    # TODO: terrible names, fix
    image = source or source_image

    pipe = CodeFormer(upscale=stage.outscale).to(device.torch_device())
    return pipe(image)
