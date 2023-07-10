from logging import getLogger
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image

from ..diffusers.load import load_pipeline
from ..diffusers.utils import encode_prompt, get_latents_from_seed, get_tile_latents, parse_prompt
from ..params import ImageParams, Size, SizeChart, StageParams
from ..server import ServerContext
from ..worker import ProgressCallback, WorkerContext
from .stage import BaseStage

logger = getLogger(__name__)


class SourceTxt2ImgStage(BaseStage):
    max_tile = SizeChart.unlimited

    def run(
        self,
        job: WorkerContext,
        server: ServerContext,
        _stage: StageParams,
        params: ImageParams,
        _source: Image.Image,
        *,
        dims: Tuple[int, int, int],
        size: Size,
        callback: Optional[ProgressCallback] = None,
        latents: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Image.Image:
        params = params.with_args(**kwargs)
        size = size.with_args(**kwargs)

        logger.info(
            "generating image using txt2img, %s steps: %s", params.steps, params.prompt
        )

        if "stage_source" in kwargs:
            logger.warn(
                "a source image was passed to a txt2img stage, and will be discarded"
            )

        prompt_pairs, loras, inversions, (prompt, negative_prompt) = parse_prompt(
            params
        )

        tile_size = params.tiles
        latent_size = size.min(tile_size, tile_size)

        # generate new latents or slice existing
        if latents is None:
            latents = get_latents_from_seed(params.seed, latent_size, params.batch)
        else:
            latents = get_tile_latents(latents, dims, latent_size)

        pipe_type = params.get_valid_pipeline("txt2img")
        pipe = load_pipeline(
            server,
            params,
            pipe_type,
            job.get_device(),
            inversions=inversions,
            loras=loras,
        )

        if params.lpw():
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

        return result.images
