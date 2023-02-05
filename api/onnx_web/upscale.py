from logging import getLogger

from PIL import Image

from .chain import (
    ChainPipeline,
    correct_gfpgan,
    upscale_resrgan,
    upscale_stable_diffusion,
)
from .device_pool import JobContext
from .params import ImageParams, SizeChart, StageParams, UpscaleParams
from .utils import ServerContext

logger = getLogger(__name__)


def run_upscale_correction(
    job: JobContext,
    server: ServerContext,
    stage: StageParams,
    params: ImageParams,
    image: Image.Image,
    *,
    upscale: UpscaleParams,
) -> Image.Image:
    """
    This is a convenience method for a chain pipeline that will run upscaling and
    correction, based on the `upscale` params.
    """
    logger.info("running upscaling and correction pipeline")

    chain = ChainPipeline()

    if upscale.scale > 1:
        if "esrgan" in upscale.upscale_model:
            stage = StageParams(tile_size=stage.tile_size, outscale=upscale.outscale)
            chain.append((upscale_resrgan, stage, None))
        elif "stable-diffusion" in upscale.upscale_model:
            mini_tile = min(SizeChart.mini, stage.tile_size)
            stage = StageParams(tile_size=mini_tile, outscale=upscale.outscale)
            chain.append((upscale_stable_diffusion, stage, None))

    if upscale.faces:
        stage = StageParams(tile_size=stage.tile_size, outscale=1)
        chain.append((correct_gfpgan, stage, None))

    return chain(job, server, params, image, prompt=params.prompt, upscale=upscale)
