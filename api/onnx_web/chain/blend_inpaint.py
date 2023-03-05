from logging import getLogger
from typing import Callable, Optional, Tuple

import numpy as np
import torch
from diffusers import OnnxStableDiffusionInpaintPipeline
from PIL import Image

from ..diffusers.load import get_latents_from_seed, load_pipeline
from ..image import expand_image, mask_filter_none, noise_source_histogram
from ..output import save_image
from ..params import Border, ImageParams, Size, SizeChart, StageParams
from ..server import ServerContext
from ..utils import is_debug
from ..worker import ProgressCallback, WorkerContext
from .utils import process_tile_order

logger = getLogger(__name__)


def blend_inpaint(
    job: WorkerContext,
    server: ServerContext,
    stage: StageParams,
    params: ImageParams,
    source: Image.Image,
    *,
    expand: Border,
    stage_source: Optional[Image.Image] = None,
    stage_mask: Optional[Image.Image] = None,
    fill_color: str = "white",
    mask_filter: Callable = mask_filter_none,
    noise_source: Callable = noise_source_histogram,
    callback: Optional[ProgressCallback] = None,
    **kwargs,
) -> Image.Image:
    params = params.with_args(**kwargs)
    expand = expand.with_args(**kwargs)
    source = source or stage_source
    logger.info(
        "blending image using inpaint, %s steps: %s", params.steps, params.prompt
    )

    if stage_mask is None:
        # if no mask was provided, keep the full source image
        stage_mask = Image.new("RGB", source.size, "black")

    source, stage_mask, noise, _full_dims = expand_image(
        source,
        stage_mask,
        expand,
        fill=fill_color,
        noise_source=noise_source,
        mask_filter=mask_filter,
    )

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

        latents = get_latents_from_seed(params.seed, size)
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
                params.prompt,
                generator=rng,
                guidance_scale=params.cfg,
                height=size.height,
                image=tile_source,
                latents=latents,
                mask_image=tile_mask,
                negative_prompt=params.negative_prompt,
                num_inference_steps=params.steps,
                width=size.width,
                eta=params.eta,
                callback=callback,
            )
        else:
            rng = np.random.RandomState(params.seed)
            result = pipe(
                params.prompt,
                generator=rng,
                guidance_scale=params.cfg,
                height=size.height,
                image=tile_source,
                latents=latents,
                mask_image=stage_mask,
                negative_prompt=params.negative_prompt,
                num_inference_steps=params.steps,
                width=size.width,
                eta=params.eta,
                callback=callback,
            )

        return result.images[0]

    output = process_tile_order(stage.tile_order, source, SizeChart.auto, 1, [outpaint])

    logger.info("final output image size: %s", output.size)
    return output
