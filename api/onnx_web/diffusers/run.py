from logging import getLogger
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from ..chain import blend_mask, upscale_outpaint
from ..chain.utils import process_tile_order
from ..output import save_image, save_params
from ..params import (
    Border,
    HighresParams,
    ImageParams,
    Size,
    StageParams,
    TileOrder,
    UpscaleParams,
)
from ..server import ServerContext
from ..server.load import get_source_filters
from ..utils import run_gc, show_system_toast
from ..worker import WorkerContext
from ..worker.context import ProgressCallback
from .load import load_pipeline
from .upscale import run_upscale_correction
from .utils import encode_prompt, get_latents_from_seed, parse_prompt

logger = getLogger(__name__)


def run_loopback(
    job: WorkerContext,
    server: ServerContext,
    params: ImageParams,
    strength: float,
    image: Image.Image,
    progress: ProgressCallback,
    inversions: List[Tuple[str, float]],
    loras: List[Tuple[str, float]],
    pipeline: Optional[Any] = None,
) -> Image.Image:
    if params.loopback == 0:
        return image

    # load img2img pipeline once
    pipe_type = params.get_valid_pipeline("img2img")
    if pipe_type == "controlnet":
        logger.debug(
            "controlnet pipeline cannot be used for loopback, switching to img2img"
        )
        pipe_type = "img2img"

    logger.debug("using %s pipeline for loopback", pipe_type)

    pipe = pipeline or load_pipeline(
        server,
        params,
        pipe_type,
        job.get_device(),
        inversions=inversions,
        loras=loras,
    )

    def loopback_iteration(source: Image.Image):
        if pipe_type == "lpw":
            rng = torch.manual_seed(params.seed)
            result = pipe.img2img(
                source,
                params.prompt,
                generator=rng,
                guidance_scale=params.cfg,
                negative_prompt=params.negative_prompt,
                num_images_per_prompt=1,
                num_inference_steps=params.steps,
                strength=strength,
                eta=params.eta,
                callback=progress,
            )
            return result.images[0]
        else:
            rng = np.random.RandomState(params.seed)
            result = pipe(
                params.prompt,
                source,
                generator=rng,
                guidance_scale=params.cfg,
                negative_prompt=params.negative_prompt,
                num_images_per_prompt=1,
                num_inference_steps=params.steps,
                strength=strength,
                eta=params.eta,
                callback=progress,
            )
            return result.images[0]

    for _i in range(params.loopback):
        image = loopback_iteration(image)

    return image


def run_highres(
    job: WorkerContext,
    server: ServerContext,
    params: ImageParams,
    size: Size,
    upscale: UpscaleParams,
    highres: HighresParams,
    image: Image.Image,
    progress: ProgressCallback,
    inversions: List[Tuple[str, float]],
    loras: List[Tuple[str, float]],
    pipeline: Optional[Any] = None,
) -> Image.Image:
    if highres.scale <= 1:
        return image

    if upscale.faces and (
        upscale.upscale_order == "correction-both"
        or upscale.upscale_order == "correction-first"
    ):
        image = run_upscale_correction(
            job,
            server,
            StageParams(),
            params,
            image,
            upscale=upscale.with_args(
                scale=1,
                outscale=1,
            ),
            callback=progress,
        )

    # load img2img pipeline once
    pipe_type = params.get_valid_pipeline("img2img")
    logger.debug("using %s pipeline for highres", pipe_type)

    highres_pipe = pipeline or load_pipeline(
        server,
        params,
        pipe_type,
        job.get_device(),
        inversions=inversions,
        loras=loras,
    )

    def highres_tile(tile: Image.Image, dims):
        scaled_size = (size.height * highres.scale, size.width * highres.scale)

        if highres.method == "bilinear":
            logger.debug("using bilinear interpolation for highres")
            tile = tile.resize(
                scaled_size, resample=Image.Resampling.BILINEAR
            )
        elif highres.method == "lanczos":
            logger.debug("using Lanczos interpolation for highres")
            tile = tile.resize(
                scaled_size, resample=Image.Resampling.LANCZOS
            )
        else:
            logger.debug("using upscaling pipeline for highres")
            tile = run_upscale_correction(
                job,
                server,
                StageParams(),
                params,
                tile,
                upscale=upscale.with_args(
                    faces=False,
                    scale=highres.scale,
                    outscale=highres.scale,
                ),
                callback=progress,
            )

        if pipe_type == "lpw":
            rng = torch.manual_seed(params.seed)
            result = highres_pipe.img2img(
                tile,
                params.prompt,
                generator=rng,
                guidance_scale=params.cfg,
                negative_prompt=params.negative_prompt,
                num_images_per_prompt=1,
                num_inference_steps=highres.steps,
                strength=highres.strength,
                eta=params.eta,
                callback=progress,
            )
            return result.images[0]
        else:
            rng = np.random.RandomState(params.seed)
            result = highres_pipe(
                params.prompt,
                tile,
                generator=rng,
                guidance_scale=params.cfg,
                negative_prompt=params.negative_prompt,
                num_images_per_prompt=1,
                num_inference_steps=highres.steps,
                strength=highres.strength,
                eta=params.eta,
                callback=progress,
            )
            return result.images[0]

    logger.info(
        "running highres fix for %s iterations at %s scale",
        highres.iterations,
        highres.scale,
    )

    for _i in range(highres.iterations):
        image = process_tile_order(
            TileOrder.grid,
            image,
            size.height // highres.scale,
            highres.scale,
            [highres_tile],
            overlap=params.overlap,
        )

    return image


def run_txt2img_pipeline(
    job: WorkerContext,
    server: ServerContext,
    params: ImageParams,
    size: Size,
    outputs: List[str],
    upscale: UpscaleParams,
    highres: HighresParams,
) -> None:
    latents = get_latents_from_seed(params.seed, size, batch=params.batch)
    prompt_pairs, loras, inversions = parse_prompt(params)

    pipe_type = params.get_valid_pipeline("txt2img")
    logger.debug("using %s pipeline for txt2img", pipe_type)

    pipe = load_pipeline(
        server,
        params,
        pipe_type,
        job.get_device(),
        inversions=inversions,
        loras=loras,
    )
    progress = job.get_progress_callback()

    if pipe_type == "lpw":
        rng = torch.manual_seed(params.seed)
        result = pipe.text2img(
            params.prompt,
            height=size.height,
            width=size.width,
            generator=rng,
            guidance_scale=params.cfg,
            latents=latents,
            negative_prompt=params.negative_prompt,
            num_images_per_prompt=params.batch,
            num_inference_steps=params.steps,
            eta=params.eta,
            callback=progress,
        )
    else:
        # encode and record alternative prompts outside of LPW
        prompt_embeds = encode_prompt(
            pipe,
            prompt_pairs,
            num_images_per_prompt=params.batch,
            do_classifier_free_guidance=params.do_cfg(),
        )
        pipe.unet.set_prompts(prompt_embeds)

        rng = np.random.RandomState(params.seed)
        result = pipe(
            params.prompt,
            height=size.height,
            width=size.width,
            generator=rng,
            guidance_scale=params.cfg,
            latents=latents,
            negative_prompt=params.negative_prompt,
            num_images_per_prompt=params.batch,
            num_inference_steps=params.steps,
            eta=params.eta,
            callback=progress,
        )

    image_outputs = list(zip(result.images, outputs))
    del result
    del pipe

    for image, output in image_outputs:
        image = run_highres(
            job,
            server,
            params,
            size,
            upscale,
            highres,
            image,
            progress,
            inversions,
            loras,
        )

        image = run_upscale_correction(
            job,
            server,
            StageParams(),
            params,
            image,
            upscale=upscale,
            callback=progress,
        )

        dest = save_image(server, output, image)
        save_params(server, output, params, size, upscale=upscale, highres=highres)

    run_gc([job.get_device()])
    show_system_toast(f"finished txt2img job: {dest}")
    logger.info("finished txt2img job: %s", dest)


def run_img2img_pipeline(
    job: WorkerContext,
    server: ServerContext,
    params: ImageParams,
    outputs: List[str],
    upscale: UpscaleParams,
    highres: HighresParams,
    source: Image.Image,
    strength: float,
    source_filter: Optional[str] = None,
) -> None:
    prompt_pairs, loras, inversions = parse_prompt(params)

    # filter the source image
    if source_filter is not None:
        f = get_source_filters().get(source_filter, None)
        if f is not None:
            logger.debug("running source filter: %s", f.__name__)
            source = f(server, source)

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

    progress = job.get_progress_callback()
    if pipe_type == "lpw":
        logger.debug("using LPW pipeline for img2img")
        rng = torch.manual_seed(params.seed)
        result = pipe.img2img(
            source,
            params.prompt,
            generator=rng,
            guidance_scale=params.cfg,
            negative_prompt=params.negative_prompt,
            num_images_per_prompt=params.batch,
            num_inference_steps=params.steps,
            eta=params.eta,
            callback=progress,
            **pipe_params,
        )
    else:
        # encode and record alternative prompts outside of LPW
        prompt_embeds = encode_prompt(pipe, prompt_pairs, params.batch, params.do_cfg())
        pipe.unet.set_prompts(prompt_embeds)

        rng = np.random.RandomState(params.seed)
        result = pipe(
            params.prompt,
            source,
            generator=rng,
            guidance_scale=params.cfg,
            negative_prompt=params.negative_prompt,
            num_images_per_prompt=params.batch,
            num_inference_steps=params.steps,
            eta=params.eta,
            callback=progress,
            **pipe_params,
        )

    images = result.images
    if source_filter is not None and source_filter != "none":
        images.append(source)

    for image, output in zip(images, outputs):
        image = run_loopback(
            job,
            server,
            params,
            strength,
            image,
            progress,
            inversions,
            loras,
        )

        image = run_highres(
            job,
            server,
            params,
            Size(source.width, source.height),
            upscale,
            highres,
            image,
            progress,
            inversions,
            loras,
        )

        image = run_upscale_correction(
            job,
            server,
            StageParams(),
            params,
            image,
            upscale=upscale,
            callback=progress,
        )

        dest = save_image(server, output, image)
        size = Size(*source.size)
        save_params(server, output, params, size, upscale=upscale)

    run_gc([job.get_device()])
    show_system_toast(f"finished img2img job: {dest}")
    logger.info("finished img2img job: %s", dest)


def run_inpaint_pipeline(
    job: WorkerContext,
    server: ServerContext,
    params: ImageParams,
    size: Size,
    outputs: List[str],
    upscale: UpscaleParams,
    highres: HighresParams,
    source: Image.Image,
    mask: Image.Image,
    border: Border,
    noise_source: Any,
    mask_filter: Any,
    fill_color: str,
    tile_order: str,
) -> None:
    progress = job.get_progress_callback()
    stage = StageParams(tile_order=tile_order)

    _prompt_pairs, loras, inversions = parse_prompt(params)

    logger.debug("applying mask filter and generating noise source")
    image = upscale_outpaint(
        job,
        server,
        stage,
        params,
        source,
        border=border,
        stage_mask=mask,
        fill_color=fill_color,
        mask_filter=mask_filter,
        noise_source=noise_source,
        callback=progress,
    )

    image = run_highres(
        job,
        server,
        params,
        size,
        upscale,
        highres,
        image,
        progress,
        inversions,
        loras,
    )

    image = run_upscale_correction(
        job,
        server,
        stage,
        params,
        image,
        upscale=upscale,
        callback=progress,
    )

    dest = save_image(server, outputs[0], image)
    save_params(server, outputs[0], params, size, upscale=upscale, border=border)

    del image

    run_gc([job.get_device()])
    show_system_toast(f"finished inpaint job: {dest}")
    logger.info("finished inpaint job: %s", dest)


def run_upscale_pipeline(
    job: WorkerContext,
    server: ServerContext,
    params: ImageParams,
    size: Size,
    outputs: List[str],
    upscale: UpscaleParams,
    highres: HighresParams,
    source: Image.Image,
) -> None:
    progress = job.get_progress_callback()
    stage = StageParams()

    _prompt_pairs, loras, inversions = parse_prompt(params)

    image = run_upscale_correction(
        job, server, stage, params, source, upscale=upscale, callback=progress
    )

    # TODO: should this come first?
    image = run_highres(
        job,
        server,
        params,
        size,
        upscale,
        highres,
        image,
        progress,
        inversions,
        loras,
    )

    dest = save_image(server, outputs[0], image)
    save_params(server, outputs[0], params, size, upscale=upscale)

    del image

    run_gc([job.get_device()])
    show_system_toast(f"finished upscale job: {dest}")
    logger.info("finished upscale job: %s", dest)


def run_blend_pipeline(
    job: WorkerContext,
    server: ServerContext,
    params: ImageParams,
    size: Size,
    outputs: List[str],
    upscale: UpscaleParams,
    # highres: HighresParams,
    sources: List[Image.Image],
    mask: Image.Image,
) -> None:
    progress = job.get_progress_callback()
    stage = StageParams()

    image = blend_mask(
        job,
        server,
        stage,
        params,
        sources=sources,
        stage_mask=mask,
        callback=progress,
    )
    image = image.convert("RGB")

    image = run_upscale_correction(
        job, server, stage, params, image, upscale=upscale, callback=progress
    )

    dest = save_image(server, outputs[0], image)
    save_params(server, outputs[0], params, size, upscale=upscale)

    del image

    run_gc([job.get_device()])
    show_system_toast(f"finished blend job: {dest}")
    logger.info("finished blend job: %s", dest)
