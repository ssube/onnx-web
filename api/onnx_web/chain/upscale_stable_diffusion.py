from logging import getLogger
from os import path
from typing import Optional

import numpy as np
from PIL import Image

from ..diffusers.load import load_pipeline
from ..diffusers.utils import encode_prompt, parse_prompt
from ..params import ImageParams, StageParams, UpscaleParams
from ..server import ServerContext
from ..worker import ProgressCallback, WorkerContext
from .base import BaseStage
from .result import StageResult

logger = getLogger(__name__)


class UpscaleStableDiffusionStage(BaseStage):
    def run(
        self,
        worker: WorkerContext,
        server: ServerContext,
        _stage: StageParams,
        params: ImageParams,
        sources: StageResult,
        *,
        upscale: UpscaleParams,
        stage_source: Optional[Image.Image] = None,
        callback: Optional[ProgressCallback] = None,
        **kwargs,
    ) -> StageResult:
        params = params.with_args(**kwargs)
        upscale = upscale.with_args(**kwargs)
        logger.info(
            "upscaling with Stable Diffusion, %s steps: %s", params.steps, params.prompt
        )

        prompt_pairs, _loras, _inversions, (prompt, negative_prompt) = parse_prompt(
            params
        )

        pipeline = load_pipeline(
            server,
            params,
            "upscale",
            worker.get_device(),
            model=path.join(server.model_path, upscale.upscale_model),
        )
        rng = np.random.RandomState(params.seed)

        if not params.is_xl():
            prompt_embeds = encode_prompt(
                pipeline,
                prompt_pairs,
                num_images_per_prompt=params.batch,
                do_classifier_free_guidance=params.do_cfg(),
            )
            pipeline.unet.set_prompts(prompt_embeds)

        outputs = []
        for source in sources.as_image():
            result = pipeline(
                prompt,
                source,
                generator=rng,
                guidance_scale=params.cfg,
                negative_prompt=negative_prompt,
                num_inference_steps=params.steps,
                eta=params.eta,
                noise_level=upscale.denoise,
                callback=callback,
            )
            outputs.extend(result.images)

        return StageResult(images=outputs)
