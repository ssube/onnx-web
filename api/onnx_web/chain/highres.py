from logging import getLogger
from typing import Optional

from ..chain.blend_img2img import BlendImg2ImgStage
from ..chain.edit_metadata import EditMetadataStage
from ..chain.upscale import stage_upscale_correction
from ..chain.upscale_simple import UpscaleSimpleStage
from ..params import HighresParams, ImageParams, StageParams, UpscaleParams
from .pipeline import ChainPipeline

logger = getLogger(__name__)


def stage_highres(
    stage: StageParams,
    params: ImageParams,
    highres: HighresParams,
    upscale: UpscaleParams,
    chain: Optional[ChainPipeline] = None,
    prompt_index: Optional[int] = None,
) -> ChainPipeline:
    logger.info("staging highres pipeline at %s", highres.scale)

    if chain is None:
        chain = ChainPipeline()

    if not highres.enabled:
        logger.debug("highres not enabled, skipping")
        return chain

    if highres.iterations < 1:
        logger.debug("no highres iterations, skipping")
        return chain

    for i in range(highres.iterations):
        if highres.method == "upscale":
            logger.debug("using upscaling pipeline for highres")
            stage_upscale_correction(
                stage,
                params,
                upscale=upscale.with_args(
                    faces=False,
                    scale=highres.scale,
                    outscale=highres.scale,
                    upscale=True,
                ),
                chain=chain,
                overlap=params.vae_overlap,
            )
        else:
            logger.debug("using simple upscaling for highres")
            chain.stage(
                UpscaleSimpleStage(),
                stage,
                method=highres.method,
                overlap=params.vae_overlap,
                upscale=upscale.with_args(
                    scale=highres.scale, outscale=highres.scale, upscale=True
                ),
            )

        chain.stage(
            BlendImg2ImgStage(),
            stage.with_args(outscale=1),
            overlap=params.vae_overlap,
            prompt_index=prompt_index + i,
            strength=highres.strength,
        )

    # add highres parameters to the image metadata
    chain.stage(
        EditMetadataStage(),
        stage.with_args(outscale=1),
        highres=highres,
        replace_params=params,
    )

    return chain
