from logging import getLogger
from os import path
from typing import Optional

import torch
from PIL import Image

from ..diffusers.load import load_pipeline
from ..diffusers.utils import encode_prompt, parse_prompt
from ..params import ImageParams, StageParams, UpscaleParams
from ..server import ServerContext
from ..worker import ProgressCallback, WorkerContext

logger = getLogger(__name__)


class UpscaleStableDiffusionStage:
    def run(
        self,
        job: WorkerContext,
        server: ServerContext,
        _stage: StageParams,
        params: ImageParams,
        source: Image.Image,
        *,
        upscale: UpscaleParams,
        stage_source: Optional[Image.Image] = None,
        callback: Optional[ProgressCallback] = None,
        **kwargs,
    ) -> Image.Image:
        params = params.with_args(**kwargs)
        upscale = upscale.with_args(**kwargs)
        source = stage_source or source
        logger.info(
            "upscaling with Stable Diffusion, %s steps: %s", params.steps, params.prompt
        )

        prompt_pairs, _loras, _inversions = parse_prompt(params)

        pipeline = load_pipeline(
            server,
            params,
            "upscale",
            job.get_device(),
            model=path.join(server.model_path, upscale.upscale_model),
        )
        generator = torch.manual_seed(params.seed)

        prompt_embeds = encode_prompt(
            pipeline,
            prompt_pairs,
            num_images_per_prompt=params.batch,
            do_classifier_free_guidance=params.do_cfg(),
        )
        pipeline.unet.set_prompts(prompt_embeds)

        return pipeline(
            params.prompt,
            source,
            generator=generator,
            guidance_scale=params.cfg,
            negative_prompt=params.negative_prompt,
            num_inference_steps=params.steps,
            eta=params.eta,
            noise_level=upscale.denoise,
            callback=callback,
        ).images[0]
