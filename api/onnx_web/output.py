from json import dumps
from PIL import Image
from typing import Any, Optional

from .params import (
    Border,
    ImageParams,
    Size,
    UpscaleParams,
)
from .utils import (
    base_join,
    ServerContext,
)


def json_params(
    output: str,
    params: ImageParams,
    size: Size,
    upscale: Optional[UpscaleParams] = None,
    border: Optional[Border] = None,
) -> Any:
    json = {
        'output': output,
        'params': params.tojson(),
    }

    if upscale is not None and border is not None:
        size = upscale.resize(size.add_border(border))

    if upscale is not None:
        json['upscale'] = upscale.tojson()
        size = upscale.resize(size)

    if border is not None:
        json['border'] = border.tojson()
        size = size.add_border(border)

    json['size'] = size

    return


def save_image(ctx: ServerContext, output: str, image: Image.Image) -> str:
    path = base_join(ctx.output_path, output)
    image.save(path)
    return path


def save_params(
    ctx: ServerContext,
    output: str,
    params: ImageParams,
    size: Size,
    upscale: Optional[UpscaleParams] = None,
    border: Optional[Border] = None,
) -> str:
    path = base_join(ctx.output_path, '%s.json' % (output))
    json = json_params(output, params, size, upscale=upscale, border=border)
    with open(path, 'w') as f:
        f.write(dumps(json))
