from logging import getLogger
from typing import Optional

import numpy as np
import torch
from PIL import Image

from ..diffusers.load import load_pipeline
from ..diffusers.utils import encode_prompt, parse_prompt, slice_prompt
from ..params import ImageParams, SizeChart, StageParams
from ..server import ServerContext
from ..worker import ProgressCallback, WorkerContext
from .base import BaseStage
from .result import StageResult

logger = getLogger(__name__)


class BlendImg2ImgStage(BaseStage):
    max_tile = SizeChart.max

    def run(
        self,
        worker: WorkerContext,
        server: ServerContext,
        _stage: StageParams,
        params: ImageParams,
        sources: StageResult,
        *,
        strength: float,
        callback: Optional[ProgressCallback] = None,
        stage_source: Optional[Image.Image] = None,
        prompt_index: Optional[int] = None,
        **kwargs,
    ) -> StageResult:
        params = params.with_args(**kwargs)

        # multi-stage prompting
        if prompt_index is not None:
            params = params.with_args(prompt=slice_prompt(params.prompt, prompt_index))

        logger.info(
            "blending image using img2img, %s steps: %s", params.steps, params.prompt
        )

        prompt_pairs, loras, inversions, (prompt, negative_prompt) = parse_prompt(
            params
        )

        pipe_type = params.get_valid_pipeline("img2img")
        pipe = load_pipeline(
            server,
            params,
            pipe_type,
            worker.get_device(),
            embeddings=inversions,
            loras=loras,
        )

        pipe_params = {}
        if params.is_pix2pix():
            pipe_params["image_guidance_scale"] = strength
        elif params.is_control():
            pipe_params["controlnet_conditioning_scale"] = strength
        else:
            pipe_params["strength"] = strength

        outputs = []
        for source in sources.as_image():
            if params.is_lpw():
                logger.debug("using LPW pipeline for img2img")
                rng = torch.manual_seed(params.seed)
                result = pipe.img2img(
                    source,
                    prompt,
                    generator=rng,
                    guidance_scale=params.cfg,
                    negative_prompt=negative_prompt,
                    num_inference_steps=params.steps,
                    callback=callback,
                    **pipe_params,
                )
            else:
                # encode and record alternative prompts outside of LPW
                if not params.is_xl():
                    prompt_embeds = encode_prompt(
                        pipe, prompt_pairs, params.batch, params.do_cfg()
                    )
                    pipe.unet.set_prompts(prompt_embeds)

                rng = np.random.RandomState(params.seed)
                result = pipe(
                    prompt,
                    generator=rng,
                    guidance_scale=params.cfg,
                    image=source,
                    negative_prompt=negative_prompt,
                    num_inference_steps=params.steps,
                    callback=callback,
                    **pipe_params,
                )

            outputs.extend(result.images)

        return StageResult(images=outputs)

    def steps(
        self,
        params: ImageParams,
        *args,
    ) -> int:
        return params.steps  # TODO: multiply by strength

    def outputs(
        self,
        params: ImageParams,
        sources: int,
    ) -> int:
        return sources + 1
