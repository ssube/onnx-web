from logging import getLogger
from os import path
from typing import Optional

import numpy as np
from gfpgan import GFPGANer
from PIL import Image
from realesrgan import RealESRGANer

from ..device_pool import JobContext
from ..params import DeviceParams, ImageParams, StageParams, UpscaleParams
from ..utils import ServerContext, run_gc
from .upscale_resrgan import load_resrgan

logger = getLogger(__name__)


last_pipeline_instance = None
last_pipeline_params = None


def load_gfpgan(
    ctx: ServerContext, upscale: UpscaleParams, device: DeviceParams, upsampler: Optional[RealESRGANer] = None
):
    global last_pipeline_instance
    global last_pipeline_params

    if upsampler is None:
        bg_upscale = upscale.rescale(upscale.outscale)
        upsampler = load_resrgan(ctx, bg_upscale, device)

    face_path = path.join(ctx.model_path, "%s.pth" % (upscale.correction_model))

    if last_pipeline_instance is not None and face_path == last_pipeline_params:
        logger.info("reusing existing GFPGAN pipeline")
        return last_pipeline_instance

    logger.debug("loading GFPGAN model from %s", face_path)

    # TODO: find a way to pass the ONNX model to underlying architectures
    gfpgan = GFPGANer(
        model_path=face_path,
        upscale=upscale.outscale,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=upsampler,
    )

    last_pipeline_instance = gfpgan
    last_pipeline_params = face_path
    run_gc()

    return gfpgan


def correct_gfpgan(
    job: JobContext,
    server: ServerContext,
    _stage: StageParams,
    _params: ImageParams,
    source_image: Image.Image,
    *,
    upscale: UpscaleParams,
    upsampler: Optional[RealESRGANer] = None,
    **kwargs,
) -> Image.Image:
    if upscale.correction_model is None:
        logger.warn("no face model given, skipping")
        return source_image

    logger.info("correcting faces with GFPGAN model: %s", upscale.correction_model)
    device = job.get_device()
    gfpgan = load_gfpgan(server, upscale, device, upsampler=upsampler)

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
