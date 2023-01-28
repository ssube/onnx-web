from diffusers import (
    DiffusionPipeline,
    # onnx
    OnnxStableDiffusionPipeline,
    OnnxStableDiffusionImg2ImgPipeline,
    OnnxStableDiffusionInpaintPipeline,
)
from PIL import Image, ImageChops
from typing import Any, Optional

from .chain import (
    StageParams,
)
from .image import (
    expand_image,
)
from .params import (
    ImageParams,
    Border,
    Size,
)
from .upscale import (
    run_upscale_correction,
    UpscaleParams,
)
from .utils import (
    is_debug,
    base_join,
    ServerContext,
)

import gc
import numpy as np
import torch

last_pipeline_instance = None
last_pipeline_options = (None, None, None)
last_pipeline_scheduler = None


def get_latents_from_seed(seed: int, size: Size) -> np.ndarray:
    '''
    From https://www.travelneil.com/stable-diffusion-updates.html
    '''
    # 1 is batch size
    latents_shape = (1, 4, size.height // 8, size.width // 8)
    # Gotta use numpy instead of torch, because torch's randn() doesn't support DML
    rng = np.random.default_rng(seed)
    image_latents = rng.standard_normal(latents_shape).astype(np.float32)
    return image_latents


def load_pipeline(pipeline: DiffusionPipeline, model: str, provider: str, scheduler: Any, device: Optional[str] = None):
    global last_pipeline_instance
    global last_pipeline_scheduler
    global last_pipeline_options

    options = (pipeline, model, provider)
    if last_pipeline_instance != None and last_pipeline_options == options:
        print('reusing existing pipeline')
        pipe = last_pipeline_instance
    else:
        print('unloading previous pipeline')
        last_pipeline_instance = None
        last_pipeline_scheduler = None
        gc.collect()
        torch.cuda.empty_cache()

        print('loading new pipeline')
        pipe = pipeline.from_pretrained(
            model,
            provider=provider,
            safety_checker=None,
            scheduler=scheduler.from_pretrained(model, subfolder='scheduler')
        )

        if device is not None:
            pipe = pipe.to(device)

        last_pipeline_instance = pipe
        last_pipeline_options = options
        last_pipeline_scheduler = scheduler

    if last_pipeline_scheduler != scheduler:
        print('loading new scheduler')
        scheduler = scheduler.from_pretrained(
            model, subfolder='scheduler')

        if device is not None:
            scheduler = scheduler.to(device)

        pipe.scheduler = scheduler
        last_pipeline_scheduler = scheduler

    print('running garbage collection during pipeline change')
    gc.collect()

    return pipe


def run_txt2img_pipeline(
    ctx: ServerContext,
    params: ImageParams,
    size: Size,
    output: str,
    upscale: UpscaleParams
):
    pipe = load_pipeline(OnnxStableDiffusionPipeline,
                         params.model, params.provider, params.scheduler)

    latents = get_latents_from_seed(params.seed, size)
    rng = np.random.RandomState(params.seed)

    result = pipe(
        params.prompt,
        height=size.height,
        width=size.width,
        generator=rng,
        guidance_scale=params.cfg,
        latents=latents,
        negative_prompt=params.negative_prompt,
        num_inference_steps=params.steps,
    )
    image = result.images[0]
    image = run_upscale_correction(
        ctx, StageParams(), params, image, upscale=upscale)

    dest = base_join(ctx.output_path, output)
    image.save(dest)

    del image
    del result

    print('saved txt2img output: %s' % (dest))


def run_img2img_pipeline(
    ctx: ServerContext,
    params: ImageParams,
    output: str,
    upscale: UpscaleParams,
    source_image: Image,
    strength: float,
):
    pipe = load_pipeline(OnnxStableDiffusionImg2ImgPipeline,
                         params.model, params.provider, params.scheduler)

    rng = np.random.RandomState(params.seed)

    result = pipe(
        params.prompt,
        generator=rng,
        guidance_scale=params.cfg,
        image=source_image,
        negative_prompt=params.negative_prompt,
        num_inference_steps=params.steps,
        strength=strength,
    )
    image = result.images[0]
    image = run_upscale_correction(
        ctx, StageParams(), params, image, upscale=upscale)

    dest = base_join(ctx.output_path, output)
    image.save(dest)

    del image
    del result

    print('saved img2img output: %s' % (dest))


def run_inpaint_pipeline(
    ctx: ServerContext,
    stage: StageParams,
    params: ImageParams,
    size: Size,
    output: str,
    upscale: UpscaleParams,
    source_image: Image,
    mask_image: Image,
    expand: Border,
    noise_source: Any,
    mask_filter: Any,
    strength: float,
    fill_color: str,
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
        fill=fill_color,
        noise_source=noise_source,
        mask_filter=mask_filter)

    if is_debug():
        source_image.save(base_join(ctx.output_path, 'last-source.png'))
        mask_image.save(base_join(ctx.output_path, 'last-mask.png'))
        noise_image.save(base_join(ctx.output_path, 'last-noise.png'))

    result = pipe(
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
    )
    image = result.images[0]

    if image.size == source_image.size:
        image = ImageChops.blend(source_image, image, strength)
    else:
        print('output image size does not match source, skipping post-blend')

    image = run_upscale_correction(
        ctx, StageParams(), params, image, upscale=upscale)

    dest = base_join(ctx.output_path, output)
    image.save(dest)

    del image
    del result

    print('saved inpaint output: %s' % (dest))


def run_upscale_pipeline(
    ctx: ServerContext,
    params: ImageParams,
    _size: Size,
    output: str,
    upscale: UpscaleParams,
    source_image: Image
):
    image = run_upscale_correction(
        ctx, StageParams(), params, source_image, upscale=upscale)

    dest = base_join(ctx.output_path, output)
    image.save(dest)

    del image

    print('saved img2img output: %s' % (dest))
