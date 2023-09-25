from logging import getLogger
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from ..diffusers.load import load_pipeline
from ..diffusers.utils import (
    encode_prompt,
    get_latents_from_seed,
    get_tile_latents,
    parse_prompt,
    slice_prompt,
)
from ..params import ImageParams, Size, SizeChart, StageParams
from ..server import ServerContext
from ..worker import ProgressCallback, WorkerContext
from .stage import BaseStage

logger = getLogger(__name__)


class SourceTxt2ImgStage(BaseStage):
    max_tile = SizeChart.max

    def run(
        self,
        worker: WorkerContext,
        server: ServerContext,
        stage: StageParams,
        params: ImageParams,
        sources: List[Image.Image],
        *,
        dims: Tuple[int, int, int] = None,
        size: Size,
        callback: Optional[ProgressCallback] = None,
        latents: Optional[np.ndarray] = None,
        prompt_index: Optional[int] = None,
        **kwargs,
    ) -> Image.Image:
        params = params.with_args(**kwargs)
        size = size.with_args(**kwargs)

        # multi-stage prompting
        if prompt_index is not None:
            params = params.with_args(prompt=slice_prompt(params.prompt, prompt_index))

        logger.info(
            "generating image using txt2img, %s steps: %s", params.steps, params.prompt
        )

        if len(sources):
            logger.info(
                "source images were passed to a source stage, new images will be appended"
            )

        prompt_pairs, loras, inversions, (prompt, negative_prompt) = parse_prompt(
            params
        )

        if params.is_xl():
            tile_size = max(stage.tile_size, params.tiles)
        else:
            tile_size = params.tiles

        # this works for panorama as well, because tile_size is already max(tile_size, *size)
        latent_size = size.min(tile_size, tile_size)

        # generate new latents or slice existing
        if latents is None:
            latents = get_latents_from_seed(int(params.seed), latent_size, params.batch)
        else:
            latents = get_tile_latents(latents, int(params.seed), latent_size, dims)

        pipe_type = params.get_valid_pipeline("txt2img")
        pipe = load_pipeline(
            server,
            params,
            pipe_type,
            worker.get_device(),
            inversions=inversions,
            loras=loras,
        )

        if params.is_lpw():
            logger.debug("using LPW pipeline for txt2img")
            rng = torch.manual_seed(params.seed)
            result = pipe.text2img(
                prompt,
                height=latent_size.height,
                width=latent_size.width,
                generator=rng,
                guidance_scale=params.cfg,
                latents=latents,
                negative_prompt=negative_prompt,
                num_images_per_prompt=params.batch,
                num_inference_steps=params.steps,
                eta=params.eta,
                callback=callback,
            )
        else:
            # encode and record alternative prompts outside of LPW
            prompt_embeds = encode_prompt(
                pipe, prompt_pairs, params.batch, params.do_cfg()
            )

            if not params.is_xl():
                pipe.unet.set_prompts(prompt_embeds)

            rng = np.random.RandomState(params.seed)
            result = pipe(
                prompt,
                height=latent_size.height,
                width=latent_size.width,
                generator=rng,
                guidance_scale=params.cfg,
                latents=latents,
                negative_prompt=negative_prompt,
                num_images_per_prompt=params.batch,
                num_inference_steps=params.steps,
                eta=params.eta,
                callback=callback,
            )

        output = list(sources)
        output.extend(result.images)
        return output

    def steps(
        self,
        params: ImageParams,
        size: Size,
    ) -> int:
        return params.steps

    def outputs(
        self,
        params: ImageParams,
        sources: int,
    ) -> int:
        return sources + 1
