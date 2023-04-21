from logging import getLogger
from os import path
from typing import Optional

import numpy as np
from PIL import Image

from ..params import DeviceParams, ImageParams, StageParams, UpscaleParams
from ..server import ServerContext
from ..utils import run_gc
from ..worker import WorkerContext

logger = getLogger(__name__)


def load_gfpgan(
    server: ServerContext,
    _stage: StageParams,
    upscale: UpscaleParams,
    device: DeviceParams,
):
    # must be within the load function for patch to take effect
    # TODO: rewrite and remove
    from gfpgan import GFPGANer

    face_path = path.join(server.cache_path, "%s.pth" % (upscale.correction_model))
    cache_key = (face_path,)
    cache_pipe = server.cache.get("gfpgan", cache_key)

    if cache_pipe is not None:
        logger.info("reusing existing GFPGAN pipeline")
        return cache_pipe

    logger.debug("loading GFPGAN model from %s", face_path)

    # TODO: find a way to pass the ONNX model to underlying architectures
    gfpgan = GFPGANer(
        arch="clean",
        bg_upsampler=None,
        channel_multiplier=2,
        device=device.torch_str(),
        model_path=face_path,
        upscale=upscale.face_outscale,
    )

    server.cache.set("gfpgan", cache_key, gfpgan)
    run_gc([device])

    return gfpgan


def correct_gfpgan(
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

    if upscale.correction_model is None:
        logger.warn("no face model given, skipping")
        return source

    logger.info("correcting faces with GFPGAN model: %s", upscale.correction_model)
    device = job.get_device()
    gfpgan = load_gfpgan(server, stage, upscale, device)

    output = np.array(source)
    _, _, output = gfpgan.enhance(
        output,
        has_aligned=False,
        only_center_face=False,
        paste_back=True,
        weight=upscale.face_strength,
    )
    output = Image.fromarray(output, "RGB")

    return output
