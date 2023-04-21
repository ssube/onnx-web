from hashlib import sha256
from json import dumps
from logging import getLogger
from os import path
from struct import pack
from time import time
from typing import Any, List, Optional

from PIL import Image

from .params import Border, HighresParams, ImageParams, Param, Size, UpscaleParams
from .server import ServerContext
from .utils import base_join

logger = getLogger(__name__)


def hash_value(sha, param: Optional[Param]):
    if param is None:
        return
    elif isinstance(param, bool):
        sha.update(bytearray(pack("!B", param)))
    elif isinstance(param, float):
        sha.update(bytearray(pack("!f", param)))
    elif isinstance(param, int):
        sha.update(bytearray(pack("!I", param)))
    elif isinstance(param, str):
        sha.update(param.encode("utf-8"))
    else:
        logger.warn("cannot hash param: %s, %s", param, type(param))


def json_params(
    outputs: List[str],
    params: ImageParams,
    size: Size,
    upscale: Optional[UpscaleParams] = None,
    border: Optional[Border] = None,
    highres: Optional[HighresParams] = None,
) -> Any:
    json = {
        "outputs": outputs,
        "params": params.tojson(),
    }

    json["params"]["model"] = path.basename(params.model)
    json["params"]["scheduler"] = params.scheduler

    if border is not None:
        json["border"] = border.tojson()
        size = size.add_border(border)

    if highres is not None:
        json["highres"] = highres.tojson()
        size = highres.resize(size)

    if upscale is not None:
        json["upscale"] = upscale.tojson()
        size = upscale.resize(size)

    json["size"] = size.tojson()

    return json


def make_output_name(
    server: ServerContext,
    mode: str,
    params: ImageParams,
    size: Size,
    extras: Optional[List[Optional[Param]]] = None,
    count: Optional[int] = None,
) -> List[str]:
    count = count or params.batch
    now = int(time())
    sha = sha256()

    hash_value(sha, mode)
    hash_value(sha, params.model)
    hash_value(sha, params.pipeline)
    hash_value(sha, params.scheduler)
    hash_value(sha, params.prompt)
    hash_value(sha, params.negative_prompt)
    hash_value(sha, params.cfg)
    hash_value(sha, params.seed)
    hash_value(sha, params.steps)
    hash_value(sha, params.eta)
    hash_value(sha, params.batch)
    hash_value(sha, size.width)
    hash_value(sha, size.height)

    if extras is not None:
        for param in extras:
            hash_value(sha, param)

    return [
        f"{mode}_{params.seed}_{sha.hexdigest()}_{now}_{i}.{server.image_format}"
        for i in range(count)
    ]


def save_image(server: ServerContext, output: str, image: Image.Image) -> str:
    path = base_join(server.output_path, output)
    image.save(path, format=server.image_format)
    logger.debug("saved output image to: %s", path)
    return path


def save_params(
    server: ServerContext,
    output: str,
    params: ImageParams,
    size: Size,
    upscale: Optional[UpscaleParams] = None,
    border: Optional[Border] = None,
    highres: Optional[HighresParams] = None,
) -> str:
    path = base_join(server.output_path, f"{output}.json")
    json = json_params(
        output, params, size, upscale=upscale, border=border, highres=highres
    )
    with open(path, "w") as f:
        f.write(dumps(json))
        logger.debug("saved image params to: %s", path)
        return path
