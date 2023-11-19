from logging import getLogger
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image

from ..diffusers.load import load_pipeline
from ..diffusers.utils import (
    LATENT_FACTOR,
    encode_prompt,
    get_latents_from_seed,
    get_tile_latents,
    parse_prompt,
    parse_reseed,
    slice_prompt,
)
from ..params import ImageParams, Size, SizeChart, StageParams
from ..server import ServerContext
from ..worker import ProgressCallback, WorkerContext
from .base import BaseStage
from .result import StageResult

logger = getLogger(__name__)


class SourceTxt2ImgStage(BaseStage):
    max_tile = SizeChart.max

    def run(
        self,
        worker: WorkerContext,
        server: ServerContext,
        stage: StageParams,
        params: ImageParams,
        sources: StageResult,
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
            "generating image using txt2img, %s steps of %s: %s",
            params.steps,
            params.model,
            params.prompt,
        )

        if len(sources):
            logger.info(
                "source images were passed to a source stage, new images will be appended"
            )

        prompt_pairs, loras, inversions, (prompt, negative_prompt) = parse_prompt(
            params
        )

        if params.is_panorama() or params.is_xl():
            tile_size = max(stage.tile_size, params.unet_tile)
        else:
            tile_size = params.unet_tile

        # this works for panorama as well, because tile_size is already max(tile_size, *size)
        latent_size = size.min(tile_size, tile_size)

        # generate new latents or slice existing
        if latents is None:
            latents = get_latents_from_seed(int(params.seed), latent_size, params.batch)
        else:
            latents = get_tile_latents(latents, int(params.seed), latent_size, dims)

        # reseed latents as needed
        reseed_rng = np.random.RandomState(params.seed)
        prompt, reseed = parse_reseed(prompt)
        for top, left, bottom, right, region_seed in reseed:
            if region_seed == -1:
                region_seed = reseed_rng.random_integers(2**32 - 1)

            logger.debug(
                "reseed latent region: [:, :, %s:%s, %s:%s] with %s",
                top,
                left,
                bottom,
                right,
                region_seed,
            )
            latents[
                :,
                :,
                top // LATENT_FACTOR : bottom // LATENT_FACTOR,
                left // LATENT_FACTOR : right // LATENT_FACTOR,
            ] = get_latents_from_seed(
                region_seed, Size(right - left, bottom - top), params.batch
            )

        pipe_type = params.get_valid_pipeline("txt2img")
        pipe = load_pipeline(
            server,
            params,
            pipe_type,
            worker.get_device(),
            embeddings=inversions,
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
            if params.is_panorama() or params.is_xl():
                logger.debug(
                    "prompt alternatives are not supported for panorama or SDXL"
                )
            else:
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

        outputs = list(sources)
        outputs.extend(result.images)
        logger.debug("produced %s outputs", len(outputs))
        return StageResult(images=outputs)

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
