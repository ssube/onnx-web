from logging import getLogger
from typing import List, Optional, Tuple

from ..chain import (
    ChainPipeline,
    PipelineStage,
    correct_codeformer,
    correct_gfpgan,
    upscale_bsrgan,
    upscale_resrgan,
    upscale_stable_diffusion,
    upscale_swinir,
)
from ..params import ImageParams, SizeChart, StageParams, UpscaleParams

logger = getLogger(__name__)


def split_upscale(
    upscale: UpscaleParams,
) -> Tuple[Optional[UpscaleParams], UpscaleParams]:
    if upscale.faces and (
        upscale.upscale_order == "correction-both"
        or upscale.upscale_order == "correction-first"
    ):
        return (
            upscale.with_args(
                scale=1,
                outscale=1,
            ),
            upscale.with_args(
                upscale_order="correction-last",
            ),
        )
    else:
        return (
            None,
            upscale,
        )


def append_upscale_correction(
    stage: StageParams,
    params: ImageParams,
    *,
    upscale: UpscaleParams,
    chain: Optional[ChainPipeline] = None,
    pre_stages: List[PipelineStage] = None,
    post_stages: List[PipelineStage] = None,
) -> ChainPipeline:
    """
    This is a convenience method for a chain pipeline that will run upscaling and
    correction, based on the `upscale` params.
    """
    logger.info(
        "running upscaling and correction pipeline at %s:%s",
        upscale.scale,
        upscale.outscale,
    )

    if chain is None:
        chain = ChainPipeline()

    if pre_stages is not None:
        for stage, params in pre_stages:
            chain.append((stage, params))

    upscale_stage = None
    if upscale.scale > 1:
        if "bsrgan" in upscale.upscale_model:
            bsrgan_params = StageParams(
                tile_size=stage.tile_size,
                outscale=upscale.outscale,
            )
            upscale_stage = (upscale_bsrgan, bsrgan_params, None)
        elif "esrgan" in upscale.upscale_model:
            esrgan_params = StageParams(
                tile_size=stage.tile_size,
                outscale=upscale.outscale,
            )
            upscale_stage = (upscale_resrgan, esrgan_params, None)
        elif "stable-diffusion" in upscale.upscale_model:
            mini_tile = min(SizeChart.mini, stage.tile_size)
            sd_params = StageParams(tile_size=mini_tile, outscale=upscale.outscale)
            upscale_stage = (upscale_stable_diffusion, sd_params, None)
        elif "swinir" in upscale.upscale_model:
            swinir_params = StageParams(
                tile_size=stage.tile_size,
                outscale=upscale.outscale,
            )
            upscale_stage = (upscale_swinir, swinir_params, None)
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

    if post_stages is not None:
        for stage, params in post_stages:
            chain.append((stage, params))

    return chain
