from logging import getLogger
from typing import Any, List, Optional

from PIL import Image

from ..chain import (
    blend_img2img,
    blend_mask,
    source_txt2img,
    upscale_highres,
    upscale_outpaint,
)
from ..chain.base import ChainPipeline
from ..output import save_image
from ..params import (
    Border,
    HighresParams,
    ImageParams,
    Size,
    StageParams,
    UpscaleParams,
)
from ..server import ServerContext
from ..server.load import get_source_filters
from ..utils import run_gc, show_system_toast
from ..worker import WorkerContext
from .upscale import append_upscale_correction, split_upscale
from .utils import parse_prompt

logger = getLogger(__name__)


def run_txt2img_pipeline(
    job: WorkerContext,
    server: ServerContext,
    params: ImageParams,
    size: Size,
    outputs: List[str],
    upscale: UpscaleParams,
    highres: HighresParams,
) -> None:
    # prepare the chain pipeline and first stage
    chain = ChainPipeline()
    stage = StageParams()
    chain.append(
        (
            source_txt2img,
            stage,
            {
                "size": size,
            },
        )
    )

    # apply upscaling and correction, before highres
    first_upscale, after_upscale = split_upscale(upscale)
    if first_upscale:
        append_upscale_correction(
            stage,
            params,
            upscale=first_upscale,
            chain=chain,
        )

    # apply highres
    for _i in range(highres.iterations):
        chain.append(
            (
                upscale_highres,
                stage,
                {
                    "highres": highres,
                    "upscale": upscale,
                },
            )
        )

    # apply upscaling and correction, after highres
    append_upscale_correction(
        StageParams(),
        params,
        upscale=upscale,
        chain=chain,
    )

    # run and save
    image = chain(job, server, params, None)

    _prompt_pairs, loras, inversions = parse_prompt(params)
    dest = save_image(
        server,
        outputs[0],
        image,
        params,
        size,
        upscale=upscale,
        highres=highres,
        inversions=inversions,
        loras=loras,
    )

    # clean up
    run_gc([job.get_device()])

    # notify the user
    show_system_toast(f"finished txt2img job: {dest}")
    logger.info("finished txt2img job: %s", dest)


def run_img2img_pipeline(
    job: WorkerContext,
    server: ServerContext,
    params: ImageParams,
    outputs: List[str],
    upscale: UpscaleParams,
    highres: HighresParams,
    source: Image.Image,
    strength: float,
    source_filter: Optional[str] = None,
) -> None:
    # run filter on the source image
    if source_filter is not None:
        f = get_source_filters().get(source_filter, None)
        if f is not None:
            logger.debug("running source filter: %s", f.__name__)
            source = f(server, source)

    # prepare the chain pipeline and first stage
    chain = ChainPipeline()
    stage = StageParams()
    chain.append(
        (
            blend_img2img,
            stage,
            {
                "strength": strength,
            },
        )
    )

    # apply upscaling and correction, before highres
    first_upscale, after_upscale = split_upscale(upscale)
    if first_upscale:
        append_upscale_correction(
            stage,
            params,
            upscale=first_upscale,
            chain=chain,
        )

    # loopback through multiple img2img iterations
    if params.loopback > 0:
        for _i in range(params.loopback):
            chain.append(
                (
                    blend_img2img,
                    stage,
                    {
                        "strength": strength,
                    },
                )
            )

    # highres, if selected
    if highres.iterations > 0:
        for _i in range(highres.iterations):
            chain.append(
                (
                    upscale_highres,
                    stage,
                    {
                        "highres": highres,
                        "upscale": upscale,
                    },
                )
            )

    # apply upscaling and correction, after highres
    append_upscale_correction(
        stage,
        params,
        upscale=after_upscale,
        chain=chain,
    )

    # run and append the filtered source
    images = [
        chain(job, server, params, source),
    ]

    if source_filter is not None and source_filter != "none":
        images.append(source)

    # save with metadata
    _prompt_pairs, loras, inversions = parse_prompt(params)
    size = Size(*source.size)

    for image, output in zip(images, outputs):
        dest = save_image(
            server,
            output,
            image,
            params,
            size,
            upscale=upscale,
            highres=highres,
            inversions=inversions,
            loras=loras,
        )

    # clean up
    run_gc([job.get_device()])

    # notify the user
    show_system_toast(f"finished img2img job: {dest}")
    logger.info("finished img2img job: %s", dest)


def run_inpaint_pipeline(
    job: WorkerContext,
    server: ServerContext,
    params: ImageParams,
    size: Size,
    outputs: List[str],
    upscale: UpscaleParams,
    highres: HighresParams,
    source: Image.Image,
    mask: Image.Image,
    border: Border,
    noise_source: Any,
    mask_filter: Any,
    fill_color: str,
    tile_order: str,
) -> None:
    logger.debug("building inpaint pipeline")

    # set up the chain pipeline and base stage
    chain = ChainPipeline()
    stage = StageParams(tile_order=tile_order)
    chain.append(
        (
            upscale_outpaint,
            stage,
            {
                "border": border,
                "stage_mask": mask,
                "fill_color": fill_color,
                "mask_filter": mask_filter,
                "noise_source": noise_source,
            },
        )
    )

    # apply highres
    chain.append(
        (
            upscale_highres,
            stage,
            {
                "highres": highres,
                "upscale": upscale,
            },
        )
    )

    # apply upscaling and correction
    append_upscale_correction(
        stage,
        params,
        upscale=upscale,
        chain=chain,
    )

    # run and save
    image = chain(job, server, params, source)

    _prompt_pairs, loras, inversions = parse_prompt(params)
    dest = save_image(
        server,
        outputs[0],
        image,
        params,
        size,
        upscale=upscale,
        border=border,
        inversions=inversions,
        loras=loras,
    )

    # clean up
    del image
    run_gc([job.get_device()])

    # notify the user
    show_system_toast(f"finished inpaint job: {dest}")
    logger.info("finished inpaint job: %s", dest)


def run_upscale_pipeline(
    job: WorkerContext,
    server: ServerContext,
    params: ImageParams,
    size: Size,
    outputs: List[str],
    upscale: UpscaleParams,
    highres: HighresParams,
    source: Image.Image,
) -> None:
    # set up the chain pipeline, no base stage for upscaling
    chain = ChainPipeline()
    stage = StageParams()

    # apply upscaling and correction, before highres
    first_upscale, after_upscale = split_upscale(upscale)
    if first_upscale:
        append_upscale_correction(
            stage,
            params,
            upscale=first_upscale,
            chain=chain,
        )

    # apply highres
    chain.append(
        (
            upscale_highres,
            stage,
            {
                "highres": highres,
                "upscale": upscale,
            },
        )
    )

    # apply upscaling and correction, after highres
    append_upscale_correction(
        stage,
        params,
        upscale=after_upscale,
        chain=chain,
    )

    # run and save
    image = chain(job, server, params, source)
    _prompt_pairs, loras, inversions = parse_prompt(params)
    dest = save_image(
        server,
        outputs[0],
        image,
        params,
        size,
        upscale=upscale,
        inversions=inversions,
        loras=loras,
    )

    # clean up
    del image
    run_gc([job.get_device()])

    # notify the user
    show_system_toast(f"finished upscale job: {dest}")
    logger.info("finished upscale job: %s", dest)


def run_blend_pipeline(
    job: WorkerContext,
    server: ServerContext,
    params: ImageParams,
    size: Size,
    outputs: List[str],
    upscale: UpscaleParams,
    # highres: HighresParams,
    sources: List[Image.Image],
    mask: Image.Image,
) -> None:
    # set up the chain pipeline and base stage
    chain = ChainPipeline()
    stage = StageParams()
    stage.append((blend_mask, stage, None))

    # apply upscaling and correction
    append_upscale_correction(
        stage,
        params,
        upscale=upscale,
        chain=chain,
    )

    # run and save
    image = chain(job, server, params, sources[0])
    dest = save_image(server, outputs[0], image, params, size, upscale=upscale)

    # clean up
    del image
    run_gc([job.get_device()])

    # notify the user
    show_system_toast(f"finished blend job: {dest}")
    logger.info("finished blend job: %s", dest)
