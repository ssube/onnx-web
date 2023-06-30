from logging import getLogger
from typing import Any, Optional

from PIL import Image

from ..chain.base import ChainPipeline
from ..chain.img2img import blend_img2img
from ..diffusers.upscale import append_upscale_correction
from ..params import HighresParams, ImageParams, StageParams, UpscaleParams
from ..server import ServerContext
from ..worker import WorkerContext
from ..worker.context import ProgressCallback

logger = getLogger(__name__)


def upscale_highres(
    job: WorkerContext,
    server: ServerContext,
    _stage: StageParams,
    params: ImageParams,
    source: Image.Image,
    *,
    highres: HighresParams,
    upscale: UpscaleParams,
    stage_source: Optional[Image.Image] = None,
    pipeline: Optional[Any] = None,
    callback: Optional[ProgressCallback] = None,
    **kwargs,
) -> Image.Image:
    source = stage_source or source

    if highres.scale <= 1:
        return source

    chain = ChainPipeline()
    scaled_size = (source.width * highres.scale, source.height * highres.scale)

    # TODO: upscaling within the same stage prevents tiling from happening and causes OOM
    if highres.method == "bilinear":
        logger.debug("using bilinear interpolation for highres")
        source = source.resize(scaled_size, resample=Image.Resampling.BILINEAR)
    elif highres.method == "lanczos":
        logger.debug("using Lanczos interpolation for highres")
        source = source.resize(scaled_size, resample=Image.Resampling.LANCZOS)
    else:
        logger.debug("using upscaling pipeline for highres")
        append_upscale_correction(
            StageParams(),
            params,
            upscale=upscale.with_args(
                faces=False,
                scale=highres.scale,
                outscale=highres.scale,
            ),
            chain=chain,
        )

    chain.append(
        (
            blend_img2img,
            StageParams(),
            {
                "overlap": params.overlap,
                "strength": highres.strength,
            },
        )
    )

    return chain(
        job,
        server,
        params,
        source,
        callback=callback,
    )
