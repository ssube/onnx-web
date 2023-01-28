from basicsr.archs.rrdbnet_arch import RRDBNet
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionUpscalePipeline,
)
from gfpgan import GFPGANer
from os import path
from PIL import Image
from realesrgan import RealESRGANer
from typing import Optional

import numpy as np
import torch

from .chain import (
    ChainPipeline,
    StageParams,
)
from .onnx import (
    ONNXNet,
    OnnxStableDiffusionUpscalePipeline,
)
from .params import (
    ImageParams,
    Size,
    UpscaleParams,
)
from .utils import (
    ServerContext,
)


def load_resrgan(ctx: ServerContext, params: UpscaleParams, tile=0):
    '''
    TODO: cache this instance
    '''
    model_file = '%s.%s' % (params.upscale_model, params.format)
    model_path = path.join(ctx.model_path, model_file)
    if not path.isfile(model_path):
        raise Exception('Real ESRGAN model not found at %s' % model_path)

    # use ONNX acceleration, if available
    if params.format == 'onnx':
        model = ONNXNet(ctx, model_file, provider=params.provider)
    elif params.format == 'pth':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                        num_block=23, num_grow_ch=32, scale=params.scale)
        raise Exception('unknown platform %s' % params.format)

    dni_weight = None
    if params.upscale_model == 'realesr-general-x4v3' and params.denoise != 1:
        wdn_model_path = model_path.replace(
            'realesr-general-x4v3', 'realesr-general-wdn-x4v3')
        model_path = [model_path, wdn_model_path]
        dni_weight = [params.denoise, 1 - params.denoise]

    # TODO: shouldn't need the PTH file
    upsampler = RealESRGANer(
        scale=params.scale,
        model_path=path.join(ctx.model_path, '%s.pth' % params.upscale_model),
        dni_weight=dni_weight,
        model=model,
        tile=tile,
        tile_pad=params.tile_pad,
        pre_pad=params.pre_pad,
        half=params.half)

    return upsampler


def load_stable_diffusion(ctx: ServerContext, upscale: UpscaleParams):
    '''
    TODO: cache this instance
    '''
    if upscale.format == 'onnx':
        model_path = path.join(ctx.model_path, upscale.upscale_model)
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

    return pipeline


def upscale_resrgan(
    ctx: ServerContext,
    stage: StageParams,
    _params: ImageParams,
    source_image: Image.Image,
    *,
    upscale: UpscaleParams,
) -> Image:
    print('upscaling image with Real ESRGAN', upscale.scale)

    output = np.array(source_image)
    upsampler = load_resrgan(ctx, upscale, tile=stage.tile_size)

    output, _ = upsampler.enhance(output, outscale=upscale.outscale)

    output = Image.fromarray(output, 'RGB')
    print('final output image size', output.size)
    return output


def correct_gfpgan(
    ctx: ServerContext,
    _stage: StageParams,
    _params: ImageParams,
    image: Image.Image,
    *,
    upscale: UpscaleParams,
    upsampler: Optional[RealESRGANer] = None,
) -> Image:
    if upscale.correction_model is None:
        print('no face model given, skipping')
        return image

    print('correcting faces with GFPGAN model: %s' % upscale.correction_model)

    if upsampler is None:
        upsampler = load_resrgan(ctx, upscale)

    face_path = path.join(ctx.model_path, '%s.pth' %
                          (upscale.correction_model))

    # TODO: doesn't have a model param, not sure how to pass ONNX model
    face_enhancer = GFPGANer(
        model_path=face_path,
        upscale=upscale.outscale,
        arch='clean',
        channel_multiplier=2,
        bg_upsampler=upsampler)

    _, _, output = face_enhancer.enhance(
        image, has_aligned=False, only_center_face=False, paste_back=True, weight=upscale.face_strength)

    return output


def upscale_stable_diffusion(
    ctx: ServerContext,
    _stage: StageParams,
    params: ImageParams,
    source: Image.Image,
    *,
    upscale: UpscaleParams,
) -> Image:
    print('upscaling with Stable Diffusion')

    pipeline = load_stable_diffusion(ctx, upscale)
    generator = torch.manual_seed(params.seed)
    seed = generator.initial_seed()

    return pipeline(
        params.prompt,
        source,
        generator=torch.manual_seed(seed),
        num_inference_steps=params.steps,
    ).images[0]


def run_upscale_correction(
    ctx: ServerContext,
    stage: StageParams,
    params: ImageParams,
    image: Image.Image,
    *,
    upscale: UpscaleParams,
) -> Image.Image:
    print('running upscale pipeline')

    chain = ChainPipeline()
    kwargs = {'upscale': upscale}

    if upscale.scale > 1:
        if 'esrgan' in upscale.upscale_model:
            stage = StageParams(tile_size=stage.tile_size,
                                outscale=upscale.outscale)
            chain.append((upscale_resrgan, stage, kwargs))
        elif 'stable-diffusion' in upscale.upscale_model:
            mini_tile = min(128, stage.tile_size)
            stage = StageParams(tile_size=mini_tile, outscale=upscale.outscale)
            chain.append((upscale_stable_diffusion, stage, kwargs))

    if upscale.faces:
        stage = StageParams(tile_size=stage.tile_size,
                            outscale=upscale.outscale)
        chain.append((correct_gfpgan, stage, kwargs))

    return chain(ctx, params, image)
