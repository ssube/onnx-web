from hashlib import sha256
from json import dumps
from logging import getLogger
from time import time
from typing import List, Optional

from piexif import ExifIFD, ImageIFD, dump
from piexif.helper import UserComment
from PIL import Image, PngImagePlugin

from .chain.result import ImageMetadata, StageResult
from .params import ImageParams, Param, Size
from .server import ServerContext
from .utils import base_join, hash_value

logger = getLogger(__name__)


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
    job_name = make_job_name(mode, params, size, extras)

    return [
        f"{job_name}_{i}.{server.image_format}" for i in range(offset, count + offset)
    ]


def make_job_name(
    mode: str,
    params: ImageParams,
    size: Size,
    extras: Optional[List[Optional[Param]]] = None,
) -> str:
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

    return f"{mode}_{params.seed}_{sha.hexdigest()}_{now}"


def save_result(
    server: ServerContext,
    result: StageResult,
    base_name: str,
) -> List[str]:
    results = []
    for i, (image, metadata) in enumerate(zip(result.as_image(), result.metadata)):
        results.append(
            save_image(
                server,
                base_name + f"_{i}.{server.image_format}",
                image,
                metadata,
            )
        )

    return results


def save_image(
    server: ServerContext,
    output: str,
    image: Image.Image,
    metadata: ImageMetadata,
) -> str:
    path = base_join(server.output_path, output)

    if server.image_format == "png":
        exif = PngImagePlugin.PngInfo()

        if metadata is not None:
            exif.add_text("make", "onnx-web")
            exif.add_text(
                "maker note",
                dumps(metadata.tojson(server, [output])),
            )
            exif.add_text("model", server.server_version)
            exif.add_text(
                "parameters",
                metadata.to_auto1111(server, [output]),
            )

        image.save(path, format=server.image_format, pnginfo=exif)
    else:
        exif = dump(
            {
                "0th": {
                    ExifIFD.MakerNote: UserComment.dump(
                        dumps(metadata.tojson(server, [output])),
                        encoding="unicode",
                    ),
                    ExifIFD.UserComment: UserComment.dump(
                        metadata.to_auto1111(server, [output]),
                        encoding="unicode",
                    ),
                    ImageIFD.Make: "onnx-web",
                    ImageIFD.Model: server.server_version,
                }
            }
        )
        image.save(path, format=server.image_format, exif=exif)

    if metadata is not None:
        save_metadata(
            server,
            output,
        )

    logger.debug("saved output image to: %s", path)
    return path


def save_metadata(
    server: ServerContext,
    output: str,
    metadata: ImageMetadata,
) -> str:
    path = base_join(server.output_path, f"{output}.json")
    json = metadata.tojson(server, [output])
    with open(path, "w") as f:
        f.write(dumps(json))
        logger.debug("saved image params to: %s", path)
        return path
