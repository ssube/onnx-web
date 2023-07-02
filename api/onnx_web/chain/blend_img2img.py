from logging import getLogger
from typing import Optional

import numpy as np
import torch
from PIL import Image

from ..diffusers.load import load_pipeline
from ..diffusers.utils import encode_prompt, parse_prompt
from ..params import ImageParams, StageParams
from ..server import ServerContext
from ..worker import ProgressCallback, WorkerContext
from .stage import BaseStage

logger = getLogger(__name__)


class BlendImg2ImgStage(BaseStage):
    def run(
        self,
        job: WorkerContext,
        server: ServerContext,
        _stage: StageParams,
        params: ImageParams,
        source: Image.Image,
        *,
        strength: float,
        callback: Optional[ProgressCallback] = None,
        stage_source: Optional[Image.Image] = None,
        **kwargs,
    ) -> Image.Image:
        params = params.with_args(**kwargs)
        source = stage_source or source
        logger.info(
            "blending image using img2img, %s steps: %s", params.steps, params.prompt
        )

        prompt_pairs, loras, inversions = parse_prompt(params)

        pipe_type = params.get_valid_pipeline("img2img")
        pipe = load_pipeline(
            server,
            params,
            pipe_type,
            job.get_device(),
            inversions=inversions,
            loras=loras,
        )

        pipe_params = {}
        if pipe_type == "controlnet":
            pipe_params["controlnet_conditioning_scale"] = strength
        elif pipe_type == "img2img":
            pipe_params["strength"] = strength
        elif pipe_type == "panorama":
            pipe_params["strength"] = strength
        elif pipe_type == "pix2pix":
            pipe_params["image_guidance_scale"] = strength

        if params.lpw():
            logger.debug("using LPW pipeline for img2img")
            rng = torch.manual_seed(params.seed)
            result = pipe.img2img(
                params.prompt,
                generator=rng,
                guidance_scale=params.cfg,
                image=source,
                negative_prompt=params.negative_prompt,
                num_inference_steps=params.steps,
                callback=callback,
                **pipe_params,
            )
        else:
            # encode and record alternative prompts outside of LPW
            prompt_embeds = encode_prompt(
                pipe, prompt_pairs, params.batch, params.do_cfg()
            )
            pipe.unet.set_prompts(prompt_embeds)

            rng = np.random.RandomState(params.seed)
            result = pipe(
                params.prompt,
                generator=rng,
                guidance_scale=params.cfg,
                image=source,
                negative_prompt=params.negative_prompt,
                num_inference_steps=params.steps,
                callback=callback,
                **pipe_params,
            )

        output = result.images[0]

        logger.info("final output image size: %sx%s", output.width, output.height)
        return output
