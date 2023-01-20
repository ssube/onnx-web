from diffusers import (
    DiffusionPipeline,
    # onnx
    OnnxStableDiffusionPipeline,
    OnnxStableDiffusionImg2ImgPipeline,
    OnnxStableDiffusionInpaintPipeline,
)
from PIL import Image, ImageChops
from typing import Any

import gc
import numpy as np

from .image import (
    expand_image,
)
from .upscale import (
    upscale_resrgan,
    UpscaleParams,
)
from .utils import (
    is_debug,
    safer_join,
    BaseParams,
    Border,
    ServerContext,
    Size,
)

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


def load_pipeline(pipeline: DiffusionPipeline, model: str, provider: str, scheduler: Any):
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

    print('running garbage collection during pipeline change')
    gc.collect()

    return pipe


def run_txt2img_pipeline(
    ctx: ServerContext,
    params: BaseParams,
    size: Size,
    output: str,
    upscale: UpscaleParams
):
    pipe = load_pipeline(OnnxStableDiffusionPipeline,
                         params.model, params.provider, params.scheduler)

    latents = get_latents_from_seed(params.seed, size)
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

    if upscale.faces or upscale.scale > 1:
        image = upscale_resrgan(ctx, upscale, image)

    dest = safer_join(ctx.output_path, output)
    image.save(dest)

    print('saved txt2img output: %s' % (dest))


def run_img2img_pipeline(
    ctx: ServerContext,
    params: BaseParams,
    output: str,
    upscale: UpscaleParams,
    source_image: Image,
    strength: float,
):
    pipe = load_pipeline(OnnxStableDiffusionImg2ImgPipeline,
                         params.model, params.provider, params.scheduler)

    rng = np.random.RandomState(params.seed)

    image = pipe(
        params.prompt,
        generator=rng,
        guidance_scale=params.cfg,
        image=source_image,
        negative_prompt=params.negative_prompt,
        num_inference_steps=params.steps,
        strength=strength,
    ).images[0]

    if upscale.faces or upscale.scale > 1:
        image = upscale_resrgan(ctx, upscale, image)

    dest = safer_join(ctx.output_path, output)
    image.save(dest)

    print('saved img2img output: %s' % (dest))


def run_inpaint_pipeline(
    ctx: ServerContext,
    params: BaseParams,
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
        source_image.save(safer_join(ctx.output_path, 'last-source.png'))
        mask_image.save(safer_join(ctx.output_path, 'last-mask.png'))
        noise_image.save(safer_join(ctx.output_path, 'last-noise.png'))

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

    if image.size == source_image.size:
        image = ImageChops.blend(source_image, image, strength)
    else:
        print('output image size does not match source, skipping post-blend')

    if upscale.faces or upscale.scale > 1:
        image = upscale_resrgan(ctx, upscale, image)

    dest = safer_join(ctx.output_path, output)
    image.save(dest)

    print('saved inpaint output: %s' % (dest))


def run_upscale_pipeline(
    ctx: ServerContext,
    _params: BaseParams,
    _size: Size,
    output: str,
    upscale: UpscaleParams,
    source_image: Image
):
    image = upscale_resrgan(ctx, upscale, source_image)

    dest = safer_join(ctx.output_path, output)
    image.save(dest)

    print('saved img2img output: %s' % (dest))
