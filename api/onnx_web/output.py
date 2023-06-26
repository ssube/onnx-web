from hashlib import sha256
from json import dumps
from logging import getLogger
from os import path
from struct import pack
from time import time
from typing import Any, List, Optional

from piexif import ExifIFD, ImageIFD, dump
from piexif.helper import UserComment
from PIL import Image, PngImagePlugin

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

    output_size = size
    if border is not None:
        json["border"] = border.tojson()
        output_size = output_size.add_border(border)

    if highres is not None:
        json["highres"] = highres.tojson()
        output_size = highres.resize(output_size)

    if upscale is not None:
        json["upscale"] = upscale.tojson()
        output_size = upscale.resize(output_size)

    json["input_size"] = size.tojson()
    json["size"] = output_size.tojson()

    return json


def str_params(
    params: ImageParams,
    size: Size,
) -> str:
    return (
        f"{params.input_prompt}. Negative prompt: {params.input_negative_prompt}."
        f"Steps: {params.steps}, Sampler: {params.scheduler}, CFG scale: {params.cfg}, "
        f"Seed: {params.seed}, Size: {size.width}x{size.height}, Model hash: TODO, Model: {params.model}, "
        f"Version: TODO, Tool: onnx-web"
    )


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


def save_image(
    server: ServerContext,
    output: str,
    image: Image.Image,
    params: Optional[ImageParams] = None,
    size: Optional[Size] = None,
    upscale: Optional[UpscaleParams] = None,
    border: Optional[Border] = None,
    highres: Optional[HighresParams] = None,
) -> str:
    path = base_join(server.output_path, output)

    if server.image_format == "png":
        exif = PngImagePlugin.PngInfo()

        if params is not None:
            exif.add_text("Parameters", str_params(params, size))
            exif.add_text(
                "JSON Parameters",
                json_params(
                    [output],
                    params,
                    size,
                    upscale=upscale,
                    border=border,
                    highres=highres,
                ),
            )

        image.save(path, format=server.image_format, pnginfo=exif)
    else:
        exif = dump(
            {
                "0th": {
                    ExifIFD.UserComment: UserComment.dump(
                        str_params(params, size), encoding="unicode"
                    ),
                    ImageIFD.Make: "onnx-web",
                    ImageIFD.Model: "TODO",
                    # TODO: add JSON params
                }
            }
        )
        image.save(path, format=server.image_format, exif=exif)

    if params is not None:
        save_params(server, output, params, size, upscale=upscale, border=border, highres=highres)

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
