from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionUpscalePipeline,
)
from logging import getLogger
from os import path
from PIL import Image

from ..diffusion.pipeline_onnx_stable_diffusion_upscale import (
    OnnxStableDiffusionUpscalePipeline,
)
from ..params import (
    ImageParams,
    StageParams,
    UpscaleParams,
)
from ..utils import (
    ServerContext,
)

import torch

logger = getLogger(__name__)


last_pipeline_instance = None
last_pipeline_params = (None, None)


def load_stable_diffusion(ctx: ServerContext, upscale: UpscaleParams):
    global last_pipeline_instance
    global last_pipeline_params

    model_path = path.join(ctx.model_path, upscale.upscale_model)
    cache_params = (model_path, upscale.format)

    if last_pipeline_instance != None and cache_params == last_pipeline_params:
        logger.info('reusing existing Stable Diffusion upscale pipeline')
        return last_pipeline_instance

    if upscale.format == 'onnx':
        # ValueError: Pipeline <class 'onnx_web.onnx.pipeline_onnx_stable_diffusion_upscale.OnnxStableDiffusionUpscalePipeline'>
        # expected {'vae', 'unet', 'text_encoder', 'tokenizer', 'scheduler', 'low_res_scheduler'},
        # but only {'scheduler', 'tokenizer', 'text_encoder', 'unet'} were passed.
        pipeline = OnnxStableDiffusionUpscalePipeline.from_pretrained(
            model_path,
            vae=AutoencoderKL.from_pretrained(
                model_path, subfolder='vae_encoder'),
            low_res_scheduler=DDPMScheduler.from_pretrained(
                model_path, subfolder='scheduler'),
        )
    else:
        pipeline = StableDiffusionUpscalePipeline.from_pretrained(
            'stabilityai/stable-diffusion-x4-upscaler')

    last_pipeline_instance = pipeline
    last_pipeline_params = cache_params

    return pipeline


def upscale_stable_diffusion(
    ctx: ServerContext,
    _stage: StageParams,
    params: ImageParams,
    source: Image.Image,
    *,
    upscale: UpscaleParams,
    prompt: str = None,
    **kwargs,
) -> Image.Image:
    prompt = prompt or params.prompt
    logger.info('upscaling with Stable Diffusion, %s steps: %s', params.steps, prompt)

    pipeline = load_stable_diffusion(ctx, upscale)
    generator = torch.manual_seed(params.seed)
    seed = generator.initial_seed()

    return pipeline(
        params.prompt,
        source,
        generator=torch.manual_seed(seed),
        num_inference_steps=params.steps,
    ).images[0]
