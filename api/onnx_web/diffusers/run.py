from logging import getLogger
from math import ceil
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
from .utils import get_latents_from_seed, parse_prompt

logger = getLogger(__name__)


def run_txt2img_pipeline(
    worker: WorkerContext,
    server: ServerContext,
    params: ImageParams,
    size: Size,
    outputs: List[str],
    upscale: UpscaleParams,
    highres: HighresParams,
) -> None:
    # if using panorama, the pipeline will tile itself (views)
    if params.is_panorama():
        tile_size = max(params.tiles, size.width, size.height)
    else:
        tile_size = params.tiles

    # prepare the chain pipeline and first stage
    chain = ChainPipeline()
    chain.stage(
        SourceTxt2ImgStage(),
        StageParams(
            tile_size=tile_size,
        ),
        size=size,
        overlap=params.overlap,
    )

    # apply upscaling and correction, before highres
    stage = StageParams(tile_size=params.tiles)
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
    latents = get_latents_from_seed(params.seed, size, batch=params.batch)
    progress = worker.get_progress_callback()
    images = chain.run(worker, server, params, [], callback=progress, latents=latents)

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
    run_gc([worker.get_device()])

    # notify the user
    show_system_toast(f"finished txt2img job: {dest}")
    logger.info("finished txt2img job: %s", dest)


def run_img2img_pipeline(
    worker: WorkerContext,
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
    progress = worker.get_progress_callback()
    images = chain(worker, server, params, [source], callback=progress)

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
    run_gc([worker.get_device()])

    # notify the user
    show_system_toast(f"finished img2img job: {dest}")
    logger.info("finished img2img job: %s", dest)


def run_inpaint_pipeline(
    worker: WorkerContext,
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
    full_res_inpaint: bool,
    full_res_inpaint_padding: float,
) -> None:
    logger.debug("building inpaint pipeline")
    tile_size = params.tiles

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

    # check if we can do full-res inpainting if no outpainting is done
    logger.debug("border zero: %s", border.isZero())
    full_res_inpaint = full_res_inpaint and border.isZero()
    if full_res_inpaint:
        mask_left, mask_top, mask_right, mask_bottom = mask.getbbox()
        logger.debug("mask bbox: %s", mask.getbbox())
        mask_width = mask_right - mask_left
        mask_height = mask_bottom - mask_top
        # ensure we have some padding around the mask when we do the inpaint (and that the region size is even)
        adj_mask_size = (
            ceil(max(mask_width, mask_height) * full_res_inpaint_padding / 2) * 2
        )
        mask_center_x = int(round((mask_right + mask_left) / 2))
        mask_center_y = int(round((mask_bottom + mask_top) / 2))
        adj_mask_border = (
            int(mask_center_x - adj_mask_size / 2),
            int(mask_center_y - adj_mask_size / 2),
            int(mask_center_x + adj_mask_size / 2),
            int(mask_center_y + adj_mask_size / 2),
        )

        # we would like to subtract the excess width (subtract a positive) and add the deficient width (subtract a negative)
        x_adj = -max(adj_mask_border[2] - source.width, 0) - min(adj_mask_border[0], 0)
        # we would like to subtract the excess height (subtract a negative) and add the deficient height (subtract a negative)
        y_adj = -max(adj_mask_border[3] - source.height, 0) - min(adj_mask_border[1], 0)

        adj_mask_border = (
            adj_mask_border[0] + x_adj,
            adj_mask_border[1] + y_adj,
            adj_mask_border[2] + x_adj,
            adj_mask_border[3] + y_adj,
        )

        border_integrity = all(
            (
                adj_mask_border[0] >= 0,
                adj_mask_border[1] >= 0,
                adj_mask_border[2] <= source.width,
                adj_mask_border[3] <= source.height,
            )
        )
        logger.debug(
            "adjusted mask size %s, mask bounding box: %s",
            adj_mask_size,
            adj_mask_border,
        )
        if border_integrity and adj_mask_size <= tile_size:
            logger.debug("performing full-res inpainting")
            original_source = source
            source = source.crop(adj_mask_border)
            source = ImageOps.contain(source, (tile_size, tile_size))
            mask = mask.crop(adj_mask_border)
            mask = ImageOps.contain(mask, (tile_size, tile_size))
            if is_debug():
                save_image(server, "adjusted-mask.png", mask)
                save_image(server, "adjusted-source.png", source)
        else:
            logger.debug("cannot perform full-res inpaint due to size issue")
            full_res_inpaint = False

    # set up the chain pipeline and base stage
    chain = ChainPipeline()
    stage = StageParams(tile_order=tile_order, tile_size=tile_size)
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
    latents = get_latents_from_seed(params.seed, size, batch=params.batch)
    progress = worker.get_progress_callback()
    images = chain(worker, server, params, [source], callback=progress, latents=latents)

    _pairs, loras, inversions, _rest = parse_prompt(params)
    for image, output in zip(images, outputs):
        if full_res_inpaint:
            if is_debug():
                save_image(server, "adjusted-output.png", image)
            mini_image = ImageOps.contain(image, (adj_mask_size, adj_mask_size))
            image = original_source
            image.paste(mini_image, box=adj_mask_border)
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
    run_gc([worker.get_device()])

    # notify the user
    show_system_toast(f"finished inpaint job: {dest}")
    logger.info("finished inpaint job: %s", dest)


def run_upscale_pipeline(
    worker: WorkerContext,
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
    progress = worker.get_progress_callback()
    images = chain(worker, server, params, [source], callback=progress)

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
    run_gc([worker.get_device()])

    # notify the user
    show_system_toast(f"finished upscale job: {dest}")
    logger.info("finished upscale job: %s", dest)


def run_blend_pipeline(
    worker: WorkerContext,
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
    progress = worker.get_progress_callback()
    images = chain(worker, server, params, sources, callback=progress)

    for image, output in zip(images, outputs):
        dest = save_image(server, output, image, params, size, upscale=upscale)

    # clean up
    del image
    run_gc([worker.get_device()])

    # notify the user
    show_system_toast(f"finished blend job: {dest}")
    logger.info("finished blend job: %s", dest)
