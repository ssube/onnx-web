from logging import getLogger
from typing import List, Optional

import numpy as np
import torch
from PIL import Image

from ..diffusers.load import load_pipeline
from ..diffusers.utils import encode_prompt, parse_prompt, slice_prompt
from ..params import ImageParams, SizeChart, StageParams
from ..server import ServerContext
from ..worker import ProgressCallback, WorkerContext
from .stage import BaseStage

logger = getLogger(__name__)


class BlendImg2ImgStage(BaseStage):
    max_tile = SizeChart.unlimited

    def run(
        self,
        worker: WorkerContext,
        server: ServerContext,
        _stage: StageParams,
        params: ImageParams,
        sources: List[Image.Image],
        *,
        strength: float,
        callback: Optional[ProgressCallback] = None,
        stage_source: Optional[Image.Image] = None,
        **kwargs,
    ) -> List[Image.Image]:
        params = params.with_args(**kwargs)

        # highres hax
        params = params.with_args(
            prompt = slice_prompt(params.prompt, 1)
        )

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
            inversions=inversions,
            loras=loras,
        )

        pipe_params = {}
        if params.is_control():
            pipe_params["controlnet_conditioning_scale"] = strength
        elif params.is_lpw():
            pipe_params["strength"] = strength
        elif params.is_panorama():
            pipe_params["strength"] = strength
        elif pipe_type == "img2img" or pipe_type == "img2img-sdxl":
            pipe_params["strength"] = strength
        elif pipe_type == "pix2pix":
            pipe_params["image_guidance_scale"] = strength

        outputs = []
        for source in sources:
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
                prompt_embeds = encode_prompt(
                    pipe, prompt_pairs, params.batch, params.do_cfg()
                )

                if not params.is_xl():
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

        return outputs
