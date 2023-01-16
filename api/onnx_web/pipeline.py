from diffusers import (
    DiffusionPipeline,
    # onnx
    OnnxStableDiffusionPipeline,
    OnnxStableDiffusionImg2ImgPipeline,
    OnnxStableDiffusionInpaintPipeline,
)
from os import environ
from PIL import Image
from typing import Any

import numpy as np

from .image import (
    expand_image,
)
from .upscale import (
    upscale_resrgan,
)
from .utils import (
    safer_join,
    BaseParams,
    Border,
    OutputPath,
    Size,
)

last_pipeline_instance = None
last_pipeline_options = (None, None, None)
last_pipeline_scheduler = None
model_path = None

# from https://www.travelneil.com/stable-diffusion-updates.html


def get_latents_from_seed(seed: int, size: Size) -> np.ndarray:
    # 1 is batch size
    latents_shape = (1, 4, size.height // 8, size.width // 8)
    # Gotta use numpy instead of torch, because torch's randn() doesn't support DML
    rng = np.random.default_rng(seed)
    image_latents = rng.standard_normal(latents_shape).astype(np.float32)
    return image_latents


def get_model_path(model: str):
    return safer_join(model_path, model)


# TODO: hax
def set_model_path(model: str):
  global model_path
  model_path = model


def load_pipeline(pipeline: DiffusionPipeline, model: str, provider: str, scheduler):
    global last_pipeline_instance
    global last_pipeline_scheduler
    global last_pipeline_options

    options = (pipeline, model, provider)
    if last_pipeline_instance != None and last_pipeline_options == options:
        print('reusing existing pipeline')
        pipe = last_pipeline_instance
    else:
        print('loading different pipeline')
        pipe = pipeline.from_pretrained(
            model,
            provider=provider,
            safety_checker=None,
            scheduler=scheduler.from_pretrained(model, subfolder='scheduler')
        )
        last_pipeline_instance = pipe
        last_pipeline_options = options
        last_pipeline_scheduler = scheduler

    if last_pipeline_scheduler != scheduler:
        print('changing pipeline scheduler')
        pipe.scheduler = scheduler.from_pretrained(
            model, subfolder='scheduler')
        last_pipeline_scheduler = scheduler

    return pipe


def run_txt2img_pipeline(params: BaseParams, size: Size, output: OutputPath):
    pipe = load_pipeline(OnnxStableDiffusionPipeline,
                         params.model, params.provider, params.scheduler)

    latents = get_latents_from_seed(params.seed, size.width, size.height)
    rng = np.random.RandomState(params.seed)

    image = pipe(
        params.prompt,
        size.width,
        size.height,
        generator=rng,
        guidance_scale=params.cfg,
        latents=latents,
        negative_prompt=params.negative_prompt,
        num_inference_steps=params.steps,
    ).images[0]
    image = upscale_resrgan(image, model_path)
    image.save(output.file)

    print('saved txt2img output: %s' % (output.file))


def run_img2img_pipeline(params: BaseParams, output: OutputPath, strength: float, input_image: Image):
    pipe = load_pipeline(OnnxStableDiffusionImg2ImgPipeline,
                         params.model, params.provider, params.scheduler)

    rng = np.random.RandomState(params.seed)

    image = pipe(
        params.prompt,
        generator=rng,
        guidance_scale=params.cfg,
        image=input_image,
        negative_prompt=params.negative_prompt,
        num_inference_steps=params.steps,
        strength=strength,
    ).images[0]
    image = upscale_resrgan(image, model_path)
    image.save(params.output.file)

    print('saved img2img output: %s' % (params.output.file))


def run_inpaint_pipeline(
    params: BaseParams,
    size: Size,
    output: OutputPath,
    source_image: Image,
    mask_image: Image,
    expand: Border,
    noise_source: Any,
    mask_filter: Any
):
    pipe = load_pipeline(OnnxStableDiffusionInpaintPipeline,
                         params.model, params.provider, params.scheduler)

    latents = get_latents_from_seed(params.seed, size)
    rng = np.random.RandomState(params.seed)

    print('applying mask filter and generating noise source')
    source_image, mask_image, noise_image, _full_dims = expand_image(
        source_image,
        mask_image,
        expand,
        noise_source=noise_source,
        mask_filter=mask_filter)

    if environ.get('DEBUG') is not None:
        source_image.save(safer_join(output.path, 'last-source.png'))
        mask_image.save(safer_join(output.path, 'last-mask.png'))
        noise_image.save(safer_join(output.path, 'last-noise.png'))

    image = pipe(
        params.prompt,
        generator=rng,
        guidance_scale=params.cfg,
        height=size.height,
        image=source_image,
        latents=latents,
        mask_image=mask_image,
        negative_prompt=params.negative_prompt,
        num_inference_steps=params.steps,
        width=size.width,
    ).images[0]

    image.save(output.file)

    print('saved inpaint output: %s' % (output.file))
