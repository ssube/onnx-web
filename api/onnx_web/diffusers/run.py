from logging import getLogger
from typing import Any, List, Optional

import numpy as np
import torch
from PIL import Image

from ..chain import blend_mask, upscale_outpaint
from ..chain.base import ChainProgress
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
from ..utils import run_gc
from ..worker import WorkerContext
from .load import get_latents_from_seed, load_pipeline
from .upscale import run_upscale_correction
from .utils import get_inversions_from_prompt, get_loras_from_prompt

logger = getLogger(__name__)


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

    (prompt, loras) = get_loras_from_prompt(params.prompt)
    (prompt, inversions) = get_inversions_from_prompt(prompt)
    params.prompt = prompt

    pipe = load_pipeline(
        server,
        "txt2img",
        params.model,
        params.scheduler,
        job.get_device(),
        inversions=inversions,
        loras=loras,
    )
    progress = job.get_progress_callback()

    if params.lpw():
        logger.debug("using LPW pipeline for txt2img")
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
        if highres.scale > 1:
            highres_progress = ChainProgress.from_progress(progress)

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
                    callback=highres_progress,
                )

            # load img2img pipeline once
            highres_pipe = load_pipeline(
                server,
                "img2img",
                params.model,
                params.scheduler,
                job.get_device(),
                inversions=inversions,
                loras=loras,
            )

            def highres_tile(tile: Image.Image, dims):
                if highres.method == "bilinear":
                    logger.debug("using bilinear interpolation for highres")
                    tile = tile.resize(
                        (size.height, size.width), resample=Image.Resampling.BILINEAR
                    )
                elif highres.method == "lanczos":
                    logger.debug("using Lanczos interpolation for highres")
                    tile = tile.resize(
                        (size.height, size.width), resample=Image.Resampling.LANCZOS
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
                        callback=highres_progress,
                    )

                if params.lpw():
                    logger.debug("using LPW pipeline for highres")
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
                        callback=highres_progress,
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
                        callback=highres_progress,
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
                    overlap=0,
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

    logger.info("finished txt2img job: %s", dest)


def run_img2img_pipeline(
    job: WorkerContext,
    server: ServerContext,
    params: ImageParams,
    outputs: List[str],
    upscale: UpscaleParams,
    source: Image.Image,
    strength: float,
    source_filter: Optional[str] = None,
) -> None:
    (prompt, loras) = get_loras_from_prompt(params.prompt)
    (prompt, inversions) = get_inversions_from_prompt(prompt)
    params.prompt = prompt

    # filter the source image
    if source_filter is not None:
        f = get_source_filters().get(source_filter, None)
        if f is not None:
            source = f(server, source)

    pipe = load_pipeline(
        server,
        params.pipeline,  # this is one of the only places this can actually vary between different pipelines
        params.model,
        params.scheduler,
        job.get_device(),
        control=params.control,
        inversions=inversions,
        loras=loras,
    )

    pipe_params = {}
    if params.pipeline == "controlnet":
        pipe_params["controlnet_conditioning_scale"] = strength
    elif params.pipeline == "img2img":
        pipe_params["strength"] = strength
    elif params.pipeline == "pix2pix":
        pipe_params["image_guidance_scale"] = strength

    progress = job.get_progress_callback()
    if params.lpw():
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
    if source_filter is not None:
        images.append(source)

    for image, output in zip(images, outputs):
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

    logger.info("finished img2img job: %s", dest)


def run_inpaint_pipeline(
    job: WorkerContext,
    server: ServerContext,
    params: ImageParams,
    size: Size,
    outputs: List[str],
    upscale: UpscaleParams,
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

    # calling the upscale_outpaint stage directly needs accumulating progress
    progress = ChainProgress.from_progress(progress)

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

    logger.info("finished inpaint job: %s", dest)


def run_upscale_pipeline(
    job: WorkerContext,
    server: ServerContext,
    params: ImageParams,
    size: Size,
    outputs: List[str],
    upscale: UpscaleParams,
    source: Image.Image,
) -> None:
    progress = job.get_progress_callback()
    stage = StageParams()

    image = run_upscale_correction(
        job, server, stage, params, source, upscale=upscale, callback=progress
    )

    dest = save_image(server, outputs[0], image)
    save_params(server, outputs[0], params, size, upscale=upscale)

    del image

    run_gc([job.get_device()])

    logger.info("finished upscale job: %s", dest)


def run_blend_pipeline(
    job: WorkerContext,
    server: ServerContext,
    params: ImageParams,
    size: Size,
    outputs: List[str],
    upscale: UpscaleParams,
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

    logger.info("finished blend job: %s", dest)
