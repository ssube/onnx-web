from logging import getLogger
from os import path
from typing import Optional

import numpy as np
from gfpgan import GFPGANer
from PIL import Image

from ..device_pool import JobContext
from ..params import DeviceParams, ImageParams, StageParams, UpscaleParams
from ..utils import ServerContext, run_gc
from .upscale_resrgan import load_resrgan

logger = getLogger(__name__)


last_pipeline_instance: Optional[GFPGANer] = None
last_pipeline_params: Optional[str] = None


def load_gfpgan(
    server: ServerContext,
    stage: StageParams,
    upscale: UpscaleParams,
    device: DeviceParams,
):
    global last_pipeline_instance
    global last_pipeline_params

    face_path = path.join(server.model_path, "%s.pth" % (upscale.correction_model))

    if last_pipeline_instance is not None and face_path == last_pipeline_params:
        logger.info("reusing existing GFPGAN pipeline")
        return last_pipeline_instance

    logger.debug("loading GFPGAN model from %s", face_path)

    upsampler = load_resrgan(server, upscale, device, tile=stage.tile_size)

    # TODO: find a way to pass the ONNX model to underlying architectures
    gfpgan = GFPGANer(
        arch="clean",
        bg_upsampler=upsampler,
        channel_multiplier=2,
        model_path=face_path,
        upscale=upscale.face_outscale,
    )

    last_pipeline_instance = gfpgan
    last_pipeline_params = face_path
    run_gc()

    return gfpgan


def correct_gfpgan(
    job: JobContext,
    server: ServerContext,
    stage: StageParams,
    _params: ImageParams,
    source_image: Image.Image,
    *,
    upscale: UpscaleParams,
    **kwargs,
) -> Image.Image:
    if upscale.correction_model is None:
        logger.warn("no face model given, skipping")
        return source_image

    logger.info("correcting faces with GFPGAN model: %s", upscale.correction_model)
    device = job.get_device()
    gfpgan = load_gfpgan(server, stage, upscale, device)

    output = np.array(source_image)
    _, _, output = gfpgan.enhance(
        output,
        has_aligned=False,
        only_center_face=False,
        paste_back=True,
        weight=upscale.face_strength,
    )
    output = Image.fromarray(output, "RGB")

    return output
