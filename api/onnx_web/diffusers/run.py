from logging import getLogger
from typing import Any, List, Optional

from PIL import Image, ImageOps

from onnx_web.chain.highres import stage_highres

from ..chain import (
    BlendImg2ImgStage,
    BlendMaskStage,
    ChainPipeline,
    SourceTxt2ImgStage,
    UpscaleOutpaintStage,
)
from ..chain.upscale import split_upscale, stage_upscale_correction
from ..image import expand_image
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
from ..utils import is_debug, run_gc, show_system_toast
from ..worker import WorkerContext
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
    stage = StageParams(
        tile_size=params.tiles,
    )
    chain.stage(
        SourceTxt2ImgStage(),
        stage,
        size=size,
        overlap=params.overlap,
    )

    # apply upscaling and correction, before highres
    first_upscale, after_upscale = split_upscale(upscale)
    if first_upscale:
        stage_upscale_correction(
            stage,
            params,
            upscale=first_upscale,
            chain=chain,
        )

    # apply highres
    stage_highres(
        stage,
        params,
        highres,
        upscale,
        chain=chain,
    )

    # apply upscaling and correction, after highres
    stage_upscale_correction(
        stage,
        params,
        upscale=after_upscale,
        chain=chain,
    )

    # run and save
    progress = job.get_progress_callback()
    images = chain(job, server, params, [], callback=progress)

    _pairs, loras, inversions, _rest = parse_prompt(params)

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
    stage = StageParams(
        tile_size=params.tiles,
    )
    chain.stage(
        BlendImg2ImgStage(),
        stage,
        strength=strength,
        overlap=params.overlap,
    )

    # apply upscaling and correction, before highres
    first_upscale, after_upscale = split_upscale(upscale)
    if first_upscale:
        stage_upscale_correction(
            stage,
            params,
            upscale=first_upscale,
            chain=chain,
        )

    # loopback through multiple img2img iterations
    for _i in range(params.loopback):
        chain.stage(
            BlendImg2ImgStage(),
            stage,
            strength=strength,
        )

    # highres, if selected
    stage_highres(
        stage,
        params,
        highres,
        upscale,
        chain=chain,
    )

    # apply upscaling and correction, after highres
    stage_upscale_correction(
        stage,
        params,
        upscale=after_upscale,
        chain=chain,
    )

    # run and append the filtered source
    progress = job.get_progress_callback()
    images = chain(job, server, params, [source], callback=progress)

    if source_filter is not None and source_filter != "none":
        images.append(source)

    # save with metadata
    _pairs, loras, inversions, _rest = parse_prompt(params)
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

    if mask is None:
        # if no mask was provided, keep the full source image
        mask = Image.new("L", source.size, 0)

    # masks start as 512x512, resize to cover the source, then trim the extra
    mask_max = max(source.width, source.height)
    mask = ImageOps.contain(mask, (mask_max, mask_max))
    mask = mask.crop((0, 0, source.width, source.height))

    source, mask, noise, full_size = expand_image(
        source,
        mask,
        border,
        fill=fill_color,
        noise_source=noise_source,
        mask_filter=mask_filter,
    )

    if is_debug():
        save_image(server, "full-source.png", source)
        save_image(server, "full-mask.png", mask)
        save_image(server, "full-noise.png", noise)

    # set up the chain pipeline and base stage
    chain = ChainPipeline()
    stage = StageParams(tile_order=tile_order, tile_size=params.tiles)
    chain.stage(
        UpscaleOutpaintStage(),
        stage,
        border=border,
        mask=mask,
        fill_color=fill_color,
        mask_filter=mask_filter,
        noise_source=noise_source,
        overlap=params.overlap,
    )

    # apply upscaling and correction, before highres
    first_upscale, after_upscale = split_upscale(upscale)
    if first_upscale:
        stage_upscale_correction(
            stage,
            params,
            upscale=first_upscale,
            chain=chain,
        )

    # apply highres
    stage_highres(
        stage,
        params,
        highres,
        upscale,
        chain=chain,
    )

    # apply upscaling and correction
    stage_upscale_correction(
        stage,
        params,
        upscale=after_upscale,
        chain=chain,
    )

    # run and save
    progress = job.get_progress_callback()
    images = chain(job, server, params, [source], callback=progress)

    _pairs, loras, inversions, _rest = parse_prompt(params)
    for image, output in zip(images, outputs):
        dest = save_image(
            server,
            output,
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
    stage = StageParams(tile_size=params.tiles)

    # apply upscaling and correction, before highres
    first_upscale, after_upscale = split_upscale(upscale)
    if first_upscale:
        stage_upscale_correction(
            stage,
            params,
            upscale=first_upscale,
            chain=chain,
        )

    # apply highres
    stage_highres(
        stage,
        params,
        highres,
        upscale,
        chain=chain,
    )

    # apply upscaling and correction, after highres
    stage_upscale_correction(
        stage,
        params,
        upscale=after_upscale,
        chain=chain,
    )

    # run and save
    progress = job.get_progress_callback()
    images = chain(job, server, params, [source], callback=progress)

    _pairs, loras, inversions, _rest = parse_prompt(params)
    for image, output in zip(images, outputs):
        dest = save_image(
            server,
            output,
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
    chain.stage(BlendMaskStage(), stage, stage_source=sources[1], stage_mask=mask)

    # apply upscaling and correction
    stage_upscale_correction(
        stage,
        params,
        upscale=upscale,
        chain=chain,
    )

    # run and save
    progress = job.get_progress_callback()
    images = chain(job, server, params, sources, callback=progress)

    for image, output in zip(images, outputs):
        dest = save_image(server, output, image, params, size, upscale=upscale)

    # clean up
    del image
    run_gc([job.get_device()])

    # notify the user
    show_system_toast(f"finished blend job: {dest}")
    logger.info("finished blend job: %s", dest)
