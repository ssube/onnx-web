from logging import getLogger

from codeformer import CodeFormer
from PIL import Image

from ..device_pool import JobContext
from ..params import ImageParams, StageParams
from ..utils import ServerContext

logger = getLogger(__name__)

pretrain_model_url = (
    "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth"
)

device = "cpu"


def correct_codeformer(
    _job: JobContext,
    _server: ServerContext,
    _stage: StageParams,
    _params: ImageParams,
    source_image: Image.Image,
    **kwargs,
) -> Image.Image:
    pipe = CodeFormer().to(device)

    return pipe(source_image)
