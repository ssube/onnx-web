from hashlib import sha256
from json import dumps
from logging import getLogger
from os import path
from struct import pack
from time import time
from typing import Any, Dict, List, Optional, Tuple

from piexif import ExifIFD, ImageIFD, dump
from piexif.helper import UserComment
from PIL import Image, PngImagePlugin

from onnx_web.convert.utils import resolve_tensor
from onnx_web.server.load import get_extra_hashes

from .params import Border, HighresParams, ImageParams, Param, Size, UpscaleParams
from .server import ServerContext
from .utils import base_join

logger = getLogger(__name__)

HASH_BUFFER_SIZE = 2**22  # 4MB


def hash_file(name: str):
    sha = sha256()
    with open(name, "rb") as f:
        while True:
            data = f.read(HASH_BUFFER_SIZE)
            if not data:
                break

            sha.update(data)

    return sha.hexdigest()


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
        logger.warning("cannot hash param: %s, %s", param, type(param))


def json_params(
    outputs: List[str],
    params: ImageParams,
    size: Size,
    upscale: Optional[UpscaleParams] = None,
    border: Optional[Border] = None,
    highres: Optional[HighresParams] = None,
    parent: Optional[Dict] = None,
) -> Any:
    json = {
        "input_size": size.tojson(),
        "outputs": outputs,
        "params": params.tojson(),
    }

    json["params"]["model"] = path.basename(params.model)
    json["params"]["scheduler"] = params.scheduler

    # calculate final output size
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

    json["size"] = output_size.tojson()

    return json


def str_params(
    server: ServerContext,
    params: ImageParams,
    size: Size,
    inversions: List[Tuple[str, float]] = None,
    loras: List[Tuple[str, float]] = None,
) -> str:
    model_name = path.basename(path.normpath(params.model))
    logger.debug("getting model hash for %s", model_name)

    model_hash = get_extra_hashes().get(model_name, None)
    if model_hash is None:
        model_hash_path = path.join(params.model, "hash.txt")
        if path.exists(model_hash_path):
            with open(model_hash_path, "r") as f:
                model_hash = f.readline().rstrip(",. \n\t\r")

    model_hash = model_hash or "unknown"
    hash_map = {
        model_name: model_hash,
    }

    inversion_hashes = ""
    if inversions is not None:
        inversion_pairs = [
            (
                name,
                hash_file(
                    resolve_tensor(path.join(server.model_path, "inversion", name))
                ).upper(),
            )
            for name, _weight in inversions
        ]
        inversion_hashes = ",".join(
            [f"{name}: {hash}" for name, hash in inversion_pairs]
        )
        hash_map.update(dict(inversion_pairs))

    lora_hashes = ""
    if loras is not None:
        lora_pairs = [
            (
                name,
                hash_file(
                    resolve_tensor(path.join(server.model_path, "lora", name))
                ).upper(),
            )
            for name, _weight in loras
        ]
        lora_hashes = ",".join([f"{name}: {hash}" for name, hash in lora_pairs])
        hash_map.update(dict(lora_pairs))

    return (
        f"{params.prompt or ''}\nNegative prompt: {params.negative_prompt or ''}\n"
        f"Steps: {params.steps}, Sampler: {params.scheduler}, CFG scale: {params.cfg}, "
        f"Seed: {params.seed}, Size: {size.width}x{size.height}, "
        f"Model hash: {model_hash}, Model: {model_name}, "
        f"Tool: onnx-web, Version: {server.server_version}, "
        f'Inversion hashes: "{inversion_hashes}", '
        f'Lora hashes: "{lora_hashes}", '
        f"Hashes: {dumps(hash_map)}"
    )


def make_output_name(
    server: ServerContext,
    mode: str,
    params: ImageParams,
    size: Size,
    extras: Optional[List[Optional[Param]]] = None,
    count: Optional[int] = None,
    offset: int = 0,
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
        for i in range(offset, count + offset)
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
    inversions: List[Tuple[str, float]] = None,
    loras: List[Tuple[str, float]] = None,
) -> str:
    path = base_join(server.output_path, output)

    if server.image_format == "png":
        exif = PngImagePlugin.PngInfo()

        if params is not None:
            exif.add_text("make", "onnx-web")
            exif.add_text(
                "maker note",
                dumps(
                    json_params(
                        [output],
                        params,
                        size,
                        upscale=upscale,
                        border=border,
                        highres=highres,
                    )
                ),
            )
            exif.add_text("model", server.server_version)
            exif.add_text(
                "parameters",
                str_params(server, params, size, inversions=inversions, loras=loras),
            )

        image.save(path, format=server.image_format, pnginfo=exif)
    else:
        exif = dump(
            {
                "0th": {
                    ExifIFD.MakerNote: UserComment.dump(
                        dumps(
                            json_params(
                                [output],
                                params,
                                size,
                                upscale=upscale,
                                border=border,
                                highres=highres,
                            )
                        ),
                        encoding="unicode",
                    ),
                    ExifIFD.UserComment: UserComment.dump(
                        str_params(
                            server, params, size, inversions=inversions, loras=loras
                        ),
                        encoding="unicode",
                    ),
                    ImageIFD.Make: "onnx-web",
                    ImageIFD.Model: server.server_version,
                }
            }
        )
        image.save(path, format=server.image_format, exif=exif)

    if params is not None:
        save_params(
            server,
            output,
            params,
            size,
            upscale=upscale,
            border=border,
            highres=highres,
        )

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
