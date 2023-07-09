from logging import getLogger
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps

from ..diffusers.load import load_pipeline
from ..diffusers.utils import (
    encode_prompt,
    get_latents_from_seed,
    get_tile_latents,
    parse_prompt,
)
from ..image import expand_image, mask_filter_none, noise_source_histogram
from ..output import save_image
from ..params import Border, ImageParams, Size, SizeChart, StageParams
from ..server import ServerContext
from ..utils import is_debug
from ..worker import ProgressCallback, WorkerContext
from .stage import BaseStage
from .tile import complete_tile, process_tile_grid, process_tile_order

logger = getLogger(__name__)


class UpscaleOutpaintStage(BaseStage):
    max_tile = SizeChart.unlimited

    def run(
        self,
        job: WorkerContext,
        server: ServerContext,
        stage: StageParams,
        params: ImageParams,
        sources: List[Image.Image],
        tile_mask: Image.Image,
        *,
        border: Border,
        stage_source: Optional[Image.Image] = None,
        stage_mask: Optional[Image.Image] = None,
        fill_color: str = "white",
        mask_filter: Callable = mask_filter_none,
        noise_source: Callable = noise_source_histogram,
        callback: Optional[ProgressCallback] = None,
        **kwargs,
    ) -> List[Image.Image]:
        prompt_pairs, loras, inversions, (prompt, negative_prompt) = parse_prompt(
            params
        )

        pipe_type = params.get_valid_pipeline("inpaint", params.pipeline)
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
        
            save_image(server, "tile-source.png", source)
            save_image(server, "tile-mask.png", tile_mask)
            
            #if the tile mask is all black, skip processing this tile
            if not tile_mask.getbbox():
                outputs.append(source)
                continue
            
            source_width, source_height = source.size
            source_size = Size(source_width, source_height)
            tile_size = params.tiles
            if max(source_size) > tile_size:
                latent_size = Size(tile_size,tile_size)
                latents = get_latents_from_seed(params.seed, latent_size)
                pipe_width=pipe_height=tile_size
            else: 
                latent_size = Size(source_size.width,source_size.height)
                latents = get_latents_from_seed(params.seed, latent_size)
                pipe_width=source_size.width
                pipe_height=source_size.height
                
            if params.lpw():
                logger.debug("using LPW pipeline for inpaint")
                rng = torch.manual_seed(params.seed)
                result = pipe.inpaint(
                    source,
                    tile_mask,
                    prompt,
                    negative_prompt=negative_prompt,
                    height=pipe_height,
                    width=pipe_width,
                    num_inference_steps=params.steps,
                    guidance_scale=params.cfg,
                    generator=rng,
                    latents=latents,
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
                    source,
                    tile_mask,
                    negative_prompt=negative_prompt,
                    height=pipe_height,
                    width=pipe_width,
                    num_inference_steps=params.steps,
                    guidance_scale=params.cfg,
                    generator=rng,
                    latents=latents,
                    callback=callback,
                )
            
            outputs.extend(result.images)

        return outputs
