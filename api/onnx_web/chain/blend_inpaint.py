from logging import getLogger
from typing import Callable, Optional, Tuple

import numpy as np
import torch
from diffusers import OnnxStableDiffusionInpaintPipeline
from PIL import Image

from ..diffusion.load import get_latents_from_seed, load_pipeline
from ..image import expand_image, mask_filter_none, noise_source_histogram
from ..output import save_image
from ..params import Border, ImageParams, Size, SizeChart, StageParams
from ..server.device_pool import JobContext, ProgressCallback
from ..utils import ServerContext, is_debug
from .utils import process_tile_order

logger = getLogger(__name__)


def blend_inpaint(
    job: JobContext,
    server: ServerContext,
    stage: StageParams,
    params: ImageParams,
    source: Image.Image,
    *,
    expand: Border,
    mask: Optional[Image.Image] = None,
    fill_color: str = "white",
    mask_filter: Callable = mask_filter_none,
    noise_source: Callable = noise_source_histogram,
    callback: ProgressCallback = None,
    **kwargs,
) -> Image.Image:
    params = params.with_args(**kwargs)
    expand = expand.with_args(**kwargs)
    logger.info(
        "blending image using inpaint, %s steps: %s", params.steps, params.prompt
    )

    if mask is None:
        # if no mask was provided, keep the full source image
        mask = Image.new("RGB", source.size, "black")

    source, mask, noise, _full_dims = expand_image(
        source,
        mask,
        expand,
        fill=fill_color,
        noise_source=noise_source,
        mask_filter=mask_filter,
    )

    if is_debug():
        save_image(server, "last-source.png", source)
        save_image(server, "last-mask.png", mask)
        save_image(server, "last-noise.png", noise)

    def outpaint(image: Image.Image, dims: Tuple[int, int, int]):
        left, top, tile = dims
        size = Size(*image.size)
        mask = mask.crop((left, top, left + tile, top + tile))

        if is_debug():
            save_image(server, "tile-source.png", image)
            save_image(server, "tile-mask.png", mask)

        latents = get_latents_from_seed(params.seed, size)
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
                params.prompt,
                generator=rng,
                guidance_scale=params.cfg,
                height=size.height,
                image=image,
                latents=latents,
                mask=mask,
                negative_prompt=params.negative_prompt,
                num_inference_steps=params.steps,
                width=size.width,
                callback=callback,
            )
        else:
            rng = np.random.RandomState(params.seed)
            result = pipe(
                params.prompt,
                generator=rng,
                guidance_scale=params.cfg,
                height=size.height,
                image=image,
                latents=latents,
                mask=mask,
                negative_prompt=params.negative_prompt,
                num_inference_steps=params.steps,
                width=size.width,
                callback=callback,
            )

        return result.images[0]

    output = process_tile_order(
        stage.tile_order, source, SizeChart.auto, 1, [outpaint]
    )

    logger.info("final output image size", output.size)
    return output
