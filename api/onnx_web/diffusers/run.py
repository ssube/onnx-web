from logging import getLogger
from math import ceil
from typing import Any, List, Optional

from PIL import Image, ImageOps

from ..chain import (
    BlendDenoiseFastNLMeansStage,
    BlendImg2ImgStage,
    BlendMaskStage,
    ChainPipeline,
    SourceTxt2ImgStage,
    UpscaleOutpaintStage,
)
from ..chain.highres import stage_highres
from ..chain.result import StageResult
from ..chain.upscale import split_upscale, stage_upscale_correction
from ..image import expand_image
from ..output import save_image, save_result
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
from .utils import get_latents_from_seed

logger = getLogger(__name__)


def get_base_tile(params: ImageParams, size: Size) -> int:
    if params.is_panorama():
        tile = max(params.unet_tile, size.width, size.height)
        logger.debug("adjusting tile size for panorama to %s", tile)
        return tile

    return params.unet_tile


def get_highres_tile(
    server: ServerContext, params: ImageParams, highres: HighresParams, tile: int
) -> int:
    if params.is_panorama() and server.has_feature("panorama-highres"):
        return tile * highres.scale

    return params.unet_tile


def run_txt2img_pipeline(
    worker: WorkerContext,
    server: ServerContext,
    params: ImageParams,
    size: Size,
    upscale: UpscaleParams,
    highres: HighresParams,
) -> None:
    # if using panorama, the pipeline will tile itself (views)
    tile_size = get_base_tile(params, size)

    # prepare the chain pipeline and first stage
    chain = ChainPipeline()
    chain.stage(
        SourceTxt2ImgStage(),
        StageParams(
            tile_size=tile_size,
        ),
        size=size,
        prompt_index=0,
        overlap=params.vae_overlap,
    )

    # apply upscaling and correction, before highres
    highres_size = get_highres_tile(server, params, highres, tile_size)
    if params.is_panorama():
        chain.stage(
            BlendDenoiseFastNLMeansStage(),
            StageParams(tile_size=highres_size),
        )

    first_upscale, after_upscale = split_upscale(upscale)
    if first_upscale:
        stage_upscale_correction(
            StageParams(outscale=first_upscale.outscale, tile_size=highres_size),
            params,
            chain=chain,
            upscale=first_upscale,
        )

    # apply highres
    stage_highres(
        StageParams(outscale=highres.scale, tile_size=highres_size),
        params,
        highres,
        upscale,
        chain=chain,
        prompt_index=1,
    )

    # apply upscaling and correction, after highres
    stage_upscale_correction(
        StageParams(outscale=after_upscale.outscale, tile_size=highres_size),
        params,
        chain=chain,
        upscale=after_upscale,
    )

    # run and save
    latents = get_latents_from_seed(params.seed, size, batch=params.batch)
    progress = worker.get_progress_callback()
    images = chain(
        worker, server, params, StageResult.empty(), callback=progress, latents=latents
    )

    # add a thumbnail, if requested
    cover = images.as_image()[0]
    if params.thumbnail and (
        cover.width > server.thumbnail_size or cover.height > server.thumbnail_size
    ):
        thumbnail = cover.copy()
        thumbnail.thumbnail((server.thumbnail_size, server.thumbnail_size))

        images.insert_image(0, thumbnail)

    save_result(server, images, worker.job)

    # clean up
    run_gc([worker.get_device()])

    # notify the user
    show_system_toast(f"finished txt2img job: {worker.job}")
    logger.info("finished txt2img job: %s", worker.job)


def run_img2img_pipeline(
    worker: WorkerContext,
    server: ServerContext,
    params: ImageParams,
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
    tile_size = get_base_tile(params, Size(*source.size))
    chain = ChainPipeline()
    chain.stage(
        BlendImg2ImgStage(),
        StageParams(
            tile_size=tile_size,
        ),
        prompt_index=0,
        strength=strength,
        overlap=params.vae_overlap,
    )

    # apply upscaling and correction, before highres
    first_upscale, after_upscale = split_upscale(upscale)
    if first_upscale:
        stage_upscale_correction(
            StageParams(
                outscale=first_upscale.outscale,
                tile_size=tile_size,
            ),
            params,
            upscale=first_upscale,
            chain=chain,
        )

    # loopback through multiple img2img iterations
    for _i in range(params.loopback):
        chain.stage(
            BlendImg2ImgStage(),
            StageParams(
                tile_size=tile_size,
            ),
            strength=strength,
        )

    # highres, if selected
    highres_size = get_highres_tile(server, params, highres, tile_size)
    stage_highres(
        StageParams(tile_size=highres_size, outscale=highres.scale),
        params,
        highres,
        upscale,
        chain=chain,
        prompt_index=1,
    )

    # apply upscaling and correction, after highres
    stage_upscale_correction(
        StageParams(tile_size=tile_size, outscale=after_upscale.scale),
        params,
        upscale=after_upscale,
        chain=chain,
    )

    # run and append the filtered source
    progress = worker.get_progress_callback()
    images = chain(
        worker, server, params, StageResult(images=[source]), callback=progress
    )

    if source_filter is not None and source_filter != "none":
        images.push_image(source)

    save_result(server, images, worker.job)

    # clean up
    run_gc([worker.get_device()])

    # notify the user
    show_system_toast(f"finished img2img job: {worker.job}")
    logger.info("finished img2img job: %s", worker.job)


def run_inpaint_pipeline(
    worker: WorkerContext,
    server: ServerContext,
    params: ImageParams,
    size: Size,
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
    tile_size = get_base_tile(params, size)

    if mask is None:
        # if no mask was provided, keep the full source image
        mask = Image.new("L", source.size, 0)

    # masks start as 512x512, resize to cover the source, then trim the extra
    mask_max = max(source.width, source.height)
    mask = ImageOps.contain(mask, (mask_max, mask_max))
    mask = mask.crop((0, 0, source.width, source.height))

    source, mask, noise, _full_size = expand_image(
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
        bbox = mask.getbbox()
        if bbox is None:
            bbox = (0, 0, source.width, source.height)

        logger.debug("mask bounding box: %s", bbox)
        mask_left, mask_top, mask_right, mask_bottom = bbox
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
    chain.stage(
        UpscaleOutpaintStage(),
        StageParams(tile_order=tile_order, tile_size=tile_size),
        border=border,
        mask=mask,
        fill_color=fill_color,
        mask_filter=mask_filter,
        noise_source=noise_source,
        overlap=params.vae_overlap,
        prompt_index=0,
    )

    # apply upscaling and correction, before highres
    first_upscale, after_upscale = split_upscale(upscale)
    if first_upscale:
        stage_upscale_correction(
            StageParams(outscale=first_upscale.outscale, tile_size=tile_size),
            params,
            upscale=first_upscale,
            chain=chain,
        )

    # apply highres
    highres_size = get_highres_tile(server, params, highres, tile_size)
    stage_highres(
        StageParams(outscale=highres.scale, tile_size=highres_size),
        params,
        highres,
        upscale,
        chain=chain,
        prompt_index=1,
    )

    # apply upscaling and correction
    stage_upscale_correction(
        StageParams(outscale=after_upscale.outscale),
        params,
        upscale=after_upscale,
        chain=chain,
    )

    # run and save
    latents = get_latents_from_seed(params.seed, size, batch=params.batch)
    progress = worker.get_progress_callback()
    images = chain(
        worker,
        server,
        params,
        StageResult(images=[source]),
        callback=progress,
        latents=latents,
    )

    for i, image, metadata in enumerate(zip(images.as_image(), images.metadata)):
        if full_res_inpaint:
            if is_debug():
                save_image(server, "adjusted-output.png", image)

            mini_image = ImageOps.contain(image, (adj_mask_size, adj_mask_size))
            image = original_source
            image.paste(mini_image, box=adj_mask_border)

        save_image(
            server,
            f"{worker.job}_{i}.{server.image_format}",
            image,
            metadata,
        )

    # clean up
    run_gc([worker.get_device()])

    # notify the user
    show_system_toast(f"finished inpaint job: {worker.job}")
    logger.info("finished inpaint job: %s", worker.job)


def run_upscale_pipeline(
    worker: WorkerContext,
    server: ServerContext,
    params: ImageParams,
    size: Size,
    upscale: UpscaleParams,
    highres: HighresParams,
    source: Image.Image,
) -> None:
    # set up the chain pipeline, no base stage for upscaling
    chain = ChainPipeline()
    tile_size = get_base_tile(params, size)

    # apply upscaling and correction, before highres
    first_upscale, after_upscale = split_upscale(upscale)
    if first_upscale:
        stage_upscale_correction(
            StageParams(outscale=first_upscale.outscale, tile_size=tile_size),
            params,
            upscale=first_upscale,
            chain=chain,
        )

    # apply highres
    highres_size = get_highres_tile(server, params, highres, tile_size)
    stage_highres(
        StageParams(outscale=highres.scale, tile_size=highres_size),
        params,
        highres,
        upscale,
        chain=chain,
        prompt_index=0,
    )

    # apply upscaling and correction, after highres
    stage_upscale_correction(
        StageParams(outscale=after_upscale.outscale, tile_size=tile_size),
        params,
        upscale=after_upscale,
        chain=chain,
    )

    # run and save
    progress = worker.get_progress_callback()
    images = chain(
        worker, server, params, StageResult(images=[source]), callback=progress
    )

    save_result(server, images, worker.job)

    # clean up
    run_gc([worker.get_device()])

    # notify the user
    show_system_toast(f"finished upscale job: {worker.job}")
    logger.info("finished upscale job: %s", worker.job)


def run_blend_pipeline(
    worker: WorkerContext,
    server: ServerContext,
    params: ImageParams,
    size: Size,
    upscale: UpscaleParams,
    # highres: HighresParams,
    sources: List[Image.Image],
    mask: Image.Image,
) -> None:
    # set up the chain pipeline and base stage
    chain = ChainPipeline()
    tile_size = get_base_tile(params, size)

    # resize mask to match source size
    stage_source = sources[1]
    stage_mask = mask.resize(stage_source.size, Image.Resampling.BILINEAR)

    chain.stage(
        BlendMaskStage(),
        StageParams(tile_size=tile_size),
        stage_source=stage_source,
        stage_mask=stage_mask,
    )

    # apply upscaling and correction
    stage_upscale_correction(
        StageParams(outscale=upscale.outscale),
        params,
        upscale=upscale,
        chain=chain,
    )

    # run and save
    progress = worker.get_progress_callback()
    images = chain(
        worker, server, params, StageResult(images=sources), callback=progress
    )

    save_result(server, images, worker.job)

    # clean up
    run_gc([worker.get_device()])

    # notify the user
    show_system_toast(f"finished blend job: {worker.job}")
    logger.info("finished blend job: %s", worker.job)
