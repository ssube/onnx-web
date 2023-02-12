from hashlib import sha256
from json import dumps
from logging import getLogger
from os import path
from struct import pack
from time import time
from typing import Any, Optional, Tuple

from PIL import Image

from .diffusion.load import get_scheduler_name
from .params import Border, ImageParams, Param, Size, UpscaleParams
from .utils import ServerContext, base_join

logger = getLogger(__name__)


def hash_value(sha, param: Param):
    if param is None:
        return
    elif isinstance(param, float):
        sha.update(bytearray(pack("!f", param)))
    elif isinstance(param, int):
        sha.update(bytearray(pack("!I", param)))
    elif isinstance(param, str):
        sha.update(param.encode("utf-8"))
    else:
        logger.warn("cannot hash param: %s, %s", param, type(param))


def json_params(
    output: str,
    params: ImageParams,
    size: Size,
    upscale: Optional[UpscaleParams] = None,
    border: Optional[Border] = None,
) -> Any:
    json = {
        "output": output,
        "params": params.tojson(),
    }

    json["params"]["model"] = path.basename(params.model)
    json["params"]["scheduler"] = get_scheduler_name(params.scheduler)

    if upscale is not None and border is not None:
        size = upscale.resize(size.add_border(border))

    if upscale is not None:
        json["upscale"] = upscale.tojson()
        size = upscale.resize(size)

    if border is not None:
        json["border"] = border.tojson()
        size = size.add_border(border)

    json["size"] = size.tojson()

    return json


def make_output_name(
    ctx: ServerContext,
    mode: str,
    params: ImageParams,
    size: Size,
    extras: Optional[Tuple[Param]] = None,
) -> str:
    now = int(time())
    sha = sha256()

    hash_value(sha, mode)
    hash_value(sha, params.model)
    hash_value(sha, params.scheduler.__name__)
    hash_value(sha, params.prompt)
    hash_value(sha, params.negative_prompt)
    hash_value(sha, params.cfg)
    hash_value(sha, params.steps)
    hash_value(sha, params.seed)
    hash_value(sha, size.width)
    hash_value(sha, size.height)

    if extras is not None:
        for param in extras:
            hash_value(sha, param)

    return "%s_%s_%s_%s.%s" % (
        mode,
        params.seed,
        sha.hexdigest(),
        now,
        ctx.image_format,
    )


def save_image(ctx: ServerContext, output: str, image: Image.Image) -> str:
    path = base_join(ctx.output_path, output)
    image.save(path, format=ctx.image_format)
    logger.debug("saved output image to: %s", path)
    return path


def save_params(
    ctx: ServerContext,
    output: str,
    params: ImageParams,
    size: Size,
    upscale: Optional[UpscaleParams] = None,
    border: Optional[Border] = None,
) -> str:
    path = base_join(ctx.output_path, "%s.json" % (output))
    json = json_params(output, params, size, upscale=upscale, border=border)
    with open(path, "w") as f:
        f.write(dumps(json))
        logger.debug("saved image params to: %s", path)
        return path
