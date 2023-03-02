from logging import getLogger
from typing import Optional

from PIL import Image

from .chain import (
    ChainPipeline,
    correct_codeformer,
    correct_gfpgan,
    upscale_resrgan,
    upscale_stable_diffusion,
)
from .params import ImageParams, SizeChart, StageParams, UpscaleParams
from .server import ServerContext
from .worker import ProgressCallback, WorkerContext

logger = getLogger(__name__)


def run_upscale_correction(
    job: WorkerContext,
    server: ServerContext,
    stage: StageParams,
    params: ImageParams,
    image: Image.Image,
    *,
    upscale: UpscaleParams,
    callback: Optional[ProgressCallback] = None,
) -> Image.Image:
    """
    This is a convenience method for a chain pipeline that will run upscaling and
    correction, based on the `upscale` params.
    """
    logger.info("running upscaling and correction pipeline")

    chain = ChainPipeline()

    upscale_stage = None
    if upscale.scale > 1:
        if "esrgan" in upscale.upscale_model:
            esrgan_params = StageParams(
                tile_size=stage.tile_size, outscale=upscale.outscale
            )
            upscale_stage = (upscale_resrgan, esrgan_params, None)
        elif "stable-diffusion" in upscale.upscale_model:
            mini_tile = min(SizeChart.mini, stage.tile_size)
            sd_params = StageParams(tile_size=mini_tile, outscale=upscale.outscale)
            upscale_stage = (upscale_stable_diffusion, sd_params, None)
        else:
            logger.warn("unknown upscaling model: %s", upscale.upscale_model)

    correct_stage = None
    if upscale.faces:
        face_params = StageParams(
            tile_size=stage.tile_size, outscale=upscale.face_outscale
        )
        if "codeformer" in upscale.correction_model:
            correct_stage = (correct_codeformer, face_params, None)
        elif "gfpgan" in upscale.correction_model:
            correct_stage = (correct_gfpgan, face_params, None)
        else:
            logger.warn("unknown correction model: %s", upscale.correction_model)

    if upscale.upscale_order == "correction-both":
        chain.append(correct_stage)
        chain.append(upscale_stage)
        chain.append(correct_stage)
    elif upscale.upscale_order == "correction-first":
        chain.append(correct_stage)
        chain.append(upscale_stage)
    elif upscale.upscale_order == "correction-last":
        chain.append(upscale_stage)
        chain.append(correct_stage)
    else:
        logger.warn("unknown upscaling order: %s", upscale.upscale_order)

    return chain(
        job,
        server,
        params,
        image,
        prompt=params.prompt,
        upscale=upscale,
        callback=callback,
    )
