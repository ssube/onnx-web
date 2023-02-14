from logging import getLogger
from typing import Callable, Tuple

import numpy as np
import torch
from diffusers import OnnxStableDiffusionInpaintPipeline
from PIL import Image, ImageDraw

from ..diffusion.load import get_latents_from_seed, get_tile_latents, load_pipeline
from ..image import expand_image, mask_filter_none, noise_source_histogram
from ..output import save_image
from ..params import Border, ImageParams, Size, SizeChart, StageParams
from ..server.device_pool import JobContext, ProgressCallback
from ..utils import ServerContext, is_debug
from .utils import process_tile_grid, process_tile_order

logger = getLogger(__name__)


def upscale_outpaint(
    job: JobContext,
    server: ServerContext,
    stage: StageParams,
    params: ImageParams,
    source_image: Image.Image,
    *,
    border: Border,
    prompt: str = None,
    mask_image: Image.Image = None,
    fill_color: str = "white",
    mask_filter: Callable = mask_filter_none,
    noise_source: Callable = noise_source_histogram,
    callback: ProgressCallback = None,
    **kwargs,
) -> Image.Image:
    prompt = prompt or params.prompt
    logger.info("upscaling image by expanding borders: %s", border)

    margin_x = float(max(border.left, border.right))
    margin_y = float(max(border.top, border.bottom))
    overlap = min(margin_x / source_image.width, margin_y / source_image.height)

    if mask_image is None:
        # if no mask was provided, keep the full source image
        mask_image = Image.new("RGB", source_image.size, "black")

    source_image, mask_image, noise_image, full_dims = expand_image(
        source_image,
        mask_image,
        border,
        fill=fill_color,
        noise_source=noise_source,
        mask_filter=mask_filter,
    )

    draw_mask = ImageDraw.Draw(mask_image)
    full_size = Size(*full_dims)
    full_latents = get_latents_from_seed(params.seed, full_size)

    if is_debug():
        save_image(server, "last-source.png", source_image)
        save_image(server, "last-mask.png", mask_image)
        save_image(server, "last-noise.png", noise_image)

    def outpaint(image: Image.Image, dims: Tuple[int, int, int]):
        left, top, tile = dims
        size = Size(*image.size)
        mask = mask_image.crop((left, top, left + tile, top + tile))

        if is_debug():
            save_image(server, "tile-source.png", image)
            save_image(server, "tile-mask.png", mask)

        latents = get_tile_latents(full_latents, dims)
        pipe = load_pipeline(
            server,
            OnnxStableDiffusionInpaintPipeline,
            params.model,
            params.scheduler,
            job.get_device(),
            params.lpw,
        )
        if params.lpw:
            logger.debug("using LPW pipeline for inpaint")
            rng = torch.manual_seed(params.seed)
            result = pipe.inpaint(
                image,
                mask,
                prompt,
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
                prompt,
                image,
                generator=rng,
                guidance_scale=params.cfg,
                height=size.height,
                latents=latents,
                mask_image=mask,
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
        output = process_tile_grid(source_image, SizeChart.auto, 1, [outpaint])
    elif border.left == border.right and border.top == border.bottom:
        logger.debug(
            "outpainting with an even border, using spiral tiling with %s overlap",
            overlap,
        )
        output = process_tile_order(
            stage.tile_order,
            source_image,
            SizeChart.auto,
            1,
            [outpaint],
            overlap=overlap,
        )
    else:
        logger.debug("outpainting with an uneven border, using grid tiling")
        output = process_tile_grid(source_image, SizeChart.auto, 1, [outpaint])

    logger.info("final output image size: %sx%s", output.width, output.height)
    return output
