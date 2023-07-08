from logging import getLogger
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from ..diffusers.load import load_pipeline
from ..diffusers.utils import encode_prompt, get_latents_from_seed, parse_prompt
from ..image import expand_image, mask_filter_none, noise_source_histogram
from ..output import save_image
from ..params import Border, ImageParams, Size, SizeChart, StageParams
from ..server import ServerContext
from ..utils import is_debug
from ..worker import ProgressCallback, WorkerContext
from .stage import BaseStage
from .tile import process_tile_order

logger = getLogger(__name__)


class BlendInpaintStage(BaseStage):
    def run(
        self,
        job: WorkerContext,
        server: ServerContext,
        stage: StageParams,
        params: ImageParams,
        sources: List[Image.Image],
        *,
        expand: Border,
        stage_source: Optional[Image.Image] = None,
        stage_mask: Optional[Image.Image] = None,
        fill_color: str = "white",
        mask_filter: Callable = mask_filter_none,
        noise_source: Callable = noise_source_histogram,
        callback: Optional[ProgressCallback] = None,
        **kwargs,
    ) -> List[Image.Image]:
        params = params.with_args(**kwargs)
        expand = expand.with_args(**kwargs)
        logger.info(
            "blending image using inpaint, %s steps: %s", params.steps, params.prompt
        )

        prompt_pairs, loras, inversions, (prompt, negative_prompt) = parse_prompt(
            params
        )
        pipe_type = params.get_valid_pipeline("inpaint")
        pipe = load_pipeline(
            server,
            params,
            pipe_type,
            job.get_device(),
            inversions=inversions,
            loras=loras,
        )

        outputs = []
        for source in sources:
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
                if params.lpw():
                    logger.debug("using LPW pipeline for inpaint")
                    rng = torch.manual_seed(params.seed)
                    result = pipe.inpaint(
                        prompt,
                        generator=rng,
                        guidance_scale=params.cfg,
                        height=size.height,
                        image=tile_source,
                        latents=latents,
                        mask_image=tile_mask,
                        negative_prompt=negative_prompt,
                        num_inference_steps=params.steps,
                        width=size.width,
                        eta=params.eta,
                        callback=callback,
                    )
                else:
                    # encode and record alternative prompts outside of LPW
                    prompt_embeds = encode_prompt(
                        pipe, prompt_pairs, params.batch, params.do_cfg()
                    )
                    pipe.unet.set_prompts(prompt_embeds)

                    rng = np.random.RandomState(params.seed)
                    result = pipe(
                        prompt,
                        generator=rng,
                        guidance_scale=params.cfg,
                        height=size.height,
                        image=tile_source,
                        latents=latents,
                        mask_image=stage_mask,
                        negative_prompt=negative_prompt,
                        num_inference_steps=params.steps,
                        width=size.width,
                        eta=params.eta,
                        callback=callback,
                    )

                return result.images[0]

            outputs.append(
                process_tile_order(
                    stage.tile_order,
                    source,
                    SizeChart.auto,
                    1,
                    [outpaint],
                    overlap=params.overlap,
                )
            )
