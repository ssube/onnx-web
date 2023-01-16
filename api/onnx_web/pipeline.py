from diffusers import (
    DiffusionPipeline,
    # onnx
    OnnxStableDiffusionPipeline,
    OnnxStableDiffusionImg2ImgPipeline,
    OnnxStableDiffusionInpaintPipeline,
)
from os import environ
from PIL import Image
from typing import Any, Union

import numpy as np

from .image import (
    expand_image,
)
from .upscale import (
    upscale_resrgan,
)
from .utils import (
    safer_join
)

last_pipeline_instance = None
last_pipeline_options = (None, None, None)
last_pipeline_scheduler = None

# from https://www.travelneil.com/stable-diffusion-updates.html


def get_latents_from_seed(seed: int, width: int, height: int) -> np.ndarray:
    # 1 is batch size
    latents_shape = (1, 4, height // 8, width // 8)
    # Gotta use numpy instead of torch, because torch's randn() doesn't support DML
    rng = np.random.default_rng(seed)
    image_latents = rng.standard_normal(latents_shape).astype(np.float32)
    return image_latents


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


def run_txt2img_pipeline(model, provider, scheduler, prompt, negative_prompt, cfg, steps, seed, output, height, width):
    pipe = load_pipeline(OnnxStableDiffusionPipeline,
                         model, provider, scheduler)

    latents = get_latents_from_seed(seed, width, height)
    rng = np.random.RandomState(seed)

    image = pipe(
        prompt,
        height,
        width,
        generator=rng,
        guidance_scale=cfg,
        latents=latents,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
    ).images[0]
    image = upscale_resrgan(image, model_path)
    image.save(output)

    print('saved txt2img output: %s' % (output))


def run_img2img_pipeline(model, provider, scheduler, prompt, negative_prompt, cfg, steps, seed, output, strength, input_image):
    pipe = load_pipeline(OnnxStableDiffusionImg2ImgPipeline,
                         model, provider, scheduler)

    rng = np.random.RandomState(seed)

    image = pipe(
        prompt,
        generator=rng,
        guidance_scale=cfg,
        image=input_image,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        strength=strength,
    ).images[0]
    image = upscale_resrgan(image, model_path)
    image.save(output)

    print('saved img2img output: %s' % (output))


def run_inpaint_pipeline(
    model: str,
    provider: str,
    scheduler: Any,
    prompt: str,
    negative_prompt: Union[str, None],
    cfg: float,
    steps: int,
    seed: int,
    output: str,
    height: int,
    width: int,
    source_image: Image,
    mask_image: Image,
    left: int,
    right: int,
    top: int,
    bottom: int,
    noise_source: Any,
    mask_filter: Any
):
    pipe = load_pipeline(OnnxStableDiffusionInpaintPipeline,
                         model, provider, scheduler)

    latents = get_latents_from_seed(seed, width, height)
    rng = np.random.RandomState(seed)

    print('applying mask filter and generating noise source')
    source_image, mask_image, noise_image, _full_dims = expand_image(
        source_image,
        mask_image,
        (left, right, top, bottom),
        noise_source=noise_source,
        mask_filter=mask_filter)

    if environ.get('DEBUG') is not None:
        source_image.save(safer_join(output_path, 'last-source.png'))
        mask_image.save(safer_join(output_path, 'last-mask.png'))
        noise_image.save(safer_join(output_path, 'last-noise.png'))

    image = pipe(
        prompt,
        generator=rng,
        guidance_scale=cfg,
        height=height,
        image=source_image,
        latents=latents,
        mask_image=mask_image,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        width=width,
    ).images[0]

    image.save(output)

    print('saved inpaint output: %s' % (output))
