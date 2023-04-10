from logging import getLogger
from os import path
from typing import Optional

import numpy as np
from PIL import Image

from ..models.onnx import OnnxModel
from ..params import DeviceParams, ImageParams, StageParams, UpscaleParams
from ..server import ServerContext
from ..utils import run_gc
from ..worker import WorkerContext

logger = getLogger(__name__)


def load_swinir(
    server: ServerContext,
    _stage: StageParams,
    upscale: UpscaleParams,
    device: DeviceParams,
):
    # must be within the load function for patch to take effect
    model_path = path.join(server.model_path, "%s.onnx" % (upscale.upscale_model))
    cache_key = (model_path,)
    cache_pipe = server.cache.get("swinir", cache_key)

    if cache_pipe is not None:
        logger.info("reusing existing SwinIR pipeline")
        return cache_pipe

    logger.debug("loading SwinIR model from %s", model_path)

    pipe = OnnxModel(
        server,
        model_path,
        provider=device.ort_provider(),
        sess_options=device.sess_options(),
    )

    server.cache.set("swinir", cache_key, pipe)
    run_gc([device])

    return pipe


def upscale_swinir(
    job: WorkerContext,
    server: ServerContext,
    stage: StageParams,
    _params: ImageParams,
    source: Image.Image,
    *,
    upscale: UpscaleParams,
    stage_source: Optional[Image.Image] = None,
    **kwargs,
) -> Image.Image:
    upscale = upscale.with_args(**kwargs)
    source = stage_source or source

    if upscale.upscale_model is None:
        logger.warn("no correction model given, skipping")
        return source

    logger.info("correcting faces with SwinIR model: %s", upscale.upscale_model)
    device = job.get_device()
    swinir = load_swinir(server, stage, upscale, device)

    tile_size = (64, 64)
    tile_x = source.width // tile_size[0]
    tile_y = source.height // tile_size[1]

    image = np.array(source) / 255.0
    image = image[:, :, [2, 1, 0]].astype(np.float32).transpose((2, 0, 1))
    image = np.expand_dims(image, axis=0)
    logger.info("SwinIR input shape: %s", image.shape)

    scale = upscale.outscale
    dest = np.zeros(
        (image.shape[0], image.shape[1], image.shape[2] * scale, image.shape[3] * scale)
    )
    logger.info("SwinIR output shape: %s", dest.shape)

    for x in range(tile_x):
        for y in range(tile_y):
            xt = x * tile_size[0]
            yt = y * tile_size[1]

            ix1 = xt
            ix2 = xt + tile_size[0]
            iy1 = yt
            iy2 = yt + tile_size[1]
            logger.info(
                "running SwinIR on tile: (%s, %s, %s, %s) -> (%s, %s, %s, %s)",
                ix1,
                ix2,
                iy1,
                iy2,
                ix1 * scale,
                ix2 * scale,
                iy1 * scale,
                iy2 * scale,
            )

            dest[
                :,
                :,
                ix1 * scale : ix2 * scale,
                iy1 * scale : iy2 * scale,
            ] = swinir(image[:, :, ix1:ix2, iy1:iy2])

    dest = np.clip(np.squeeze(dest, axis=0), 0, 1)
    dest = dest[[2, 1, 0], :, :].transpose((1, 2, 0))
    dest = (dest * 255.0).round().astype(np.uint8)

    output = Image.fromarray(dest, "RGB")
    logger.info("output image size: %s x %s", output.width, output.height)
    return output
