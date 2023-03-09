from logging import getLogger
from typing import Callable, Optional, Tuple

import numpy as np
import torch
from diffusers import OnnxStableDiffusionInpaintPipeline
from PIL import Image, ImageDraw

from ..diffusers.load import get_latents_from_seed, get_tile_latents, load_pipeline
from ..image import expand_image, mask_filter_none, noise_source_histogram
from ..output import save_image
from ..params import Border, ImageParams, Size, SizeChart, StageParams
from ..server import ServerContext
from ..utils import is_debug
from ..worker import ProgressCallback, WorkerContext
from .utils import process_tile_grid, process_tile_order

logger = getLogger(__name__)


def upscale_outpaint(
    job: WorkerContext,
    server: ServerContext,
    stage: StageParams,
    params: ImageParams,
    source: Image.Image,
    *,
    border: Border,
    stage_source: Optional[Image.Image] = None,
    stage_mask: Optional[Image.Image] = None,
    fill_color: str = "white",
    mask_filter: Callable = mask_filter_none,
    noise_source: Callable = noise_source_histogram,
    callback: Optional[ProgressCallback] = None,
    **kwargs,
) -> Image.Image:
    source = stage_source or source
    logger.info("upscaling image by expanding borders: %s", border)

    margin_x = float(max(border.left, border.right))
    margin_y = float(max(border.top, border.bottom))
    overlap = min(margin_x / source.width, margin_y / source.height)

    if stage_mask is None:
        # if no mask was provided, keep the full source image
        stage_mask = Image.new("RGB", source.size, "black")

    source, stage_mask, noise, full_dims = expand_image(
        source,
        stage_mask,
        border,
        fill=fill_color,
        noise_source=noise_source,
        mask_filter=mask_filter,
    )

    draw_mask = ImageDraw.Draw(stage_mask)
    full_size = Size(*full_dims)
    full_latents = get_latents_from_seed(params.seed, full_size)

    if is_debug():
        save_image(server, "last-source.png", source)
        save_image(server, "last-mask.png", stage_mask)
        save_image(server, "last-noise.png", noise)

    def outpaint(tile_source: Image.Image, dims: Tuple[int, int, int]):
        left, top, tile = dims
        size = Size(*tile_source.size)
        tile_mask = stage_mask.crop((left, top, left + tile, top + tile))

        if is_debug():
            save_image(server, "tile-source.png", tile_source)
            save_image(server, "tile-mask.png", tile_mask)

        latents = get_tile_latents(full_latents, dims)
        pipe = load_pipeline(
            server,
            OnnxStableDiffusionInpaintPipeline,
            params.model,
            params.scheduler,
            job.get_device(),
            params.lpw,
            params.inversion,
        )
        if params.lpw:
            logger.debug("using LPW pipeline for inpaint")
            rng = torch.manual_seed(params.seed)
            result = pipe.inpaint(
                tile_source,
                tile_mask,
                params.prompt,
                generator=rng,
                guidance_scale=params.cfg,
                height=size.height,
                latents=latents,
                negative_prompt=params.negative_prompt,
                num_inference_steps=params.steps,
                width=size.width,
                callback=callback,
            )
        else:
            rng = np.random.RandomState(params.seed)
            result = pipe(
                params.prompt,
                tile_source,
                generator=rng,
                guidance_scale=params.cfg,
                height=size.height,
                latents=latents,
                mask_image=tile_mask,
                negative_prompt=params.negative_prompt,
                num_inference_steps=params.steps,
                width=size.width,
                callback=callback,
            )

        # once part of the image has been drawn, keep it
        draw_mask.rectangle((left, top, left + tile, top + tile), fill="black")
        return result.images[0]

    if overlap == 0:
        logger.debug("outpainting with 0 margin, using grid tiling")
        output = process_tile_grid(source, SizeChart.auto, 1, [outpaint])
    elif border.left == border.right and border.top == border.bottom:
        logger.debug(
            "outpainting with an even border, using spiral tiling with %s overlap",
            overlap,
        )
        output = process_tile_order(
            stage.tile_order,
            source,
            SizeChart.auto,
            1,
            [outpaint],
            overlap=overlap,
        )
    else:
        logger.debug("outpainting with an uneven border, using grid tiling")
        output = process_tile_grid(source, SizeChart.auto, 1, [outpaint])

    logger.info("final output image size: %sx%s", output.width, output.height)
    return output
