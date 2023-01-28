from PIL import Image


from ..params import (
    ImageParams,
    StageParams,
)
from ..utils import (
    ServerContext,
)


def persist_disk(
    ctx: ServerContext,
    stage: StageParams,
    params: ImageParams,
    source_image: Image.Image,
    *,
    output: str,
) -> Image.Image:
    source_image.save(output)
    print('saved image to %s' % (output,))
    return source_image
