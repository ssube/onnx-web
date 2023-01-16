from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from gfpgan import GFPGANer
from onnxruntime import InferenceSession
from os import path
from PIL import Image
from realesrgan import RealESRGANer
from typing import Any

import numpy as np
import torch

from .utils import (
    ServerContext
)

# TODO: these should all be params or config
fp16 = False
outscale = 4
pre_pad = 0
tile_pad = 10

gfpgan_url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth'
resrgan_name = 'RealESRGAN_x4plus'
resrgan_url = [
    'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']


class ONNXImage():
    def __init__(self, source) -> None:
        self.source = source
        self.data = self

    def __getitem__(self, *args):
        return torch.from_numpy(self.source.__getitem__(*args)).to(torch.float32)

    def squeeze(self):
        self.source = np.squeeze(self.source, (0))
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def clamp_(self, min, max):
        self.source = np.clip(self.source, min, max)
        return self

    def numpy(self):
        return self.source


class ONNXNet():
    '''
    Provides the RRDBNet interface but using ONNX.
    '''

    def __init__(self, ctx: ServerContext) -> None:
        '''
        TODO: get platform provider from request params
        '''
        model_path = path.join(ctx.model_path, resrgan_name + '.onnx')
        self.session = InferenceSession(
            model_path, providers=['DmlExecutionProvider'])

    def __call__(self, image: Any) -> Any:
        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name
        output = self.session.run([output_name], {
            input_name: image.cpu().numpy()
        })[0]
        return ONNXImage(output)

    def eval(self) -> None:
        pass

    def half(self):
        return self

    def load_state_dict(self, net, strict=True) -> None:
        pass

    def to(self, device):
        return self


class UpscaleParams():
    def __init__(self, scale=4, faces=True, platform='onnx', denoise=0.5) -> None:
        self.denoise = denoise
        self.scale = scale
        self.faces = faces
        self.platform = platform


def make_resrgan(ctx: ServerContext, params: UpscaleParams, tile=0):
    model_path = path.join(ctx.model_path, resrgan_name + '.pth')
    if not path.isfile(model_path):
        for url in resrgan_url:
            model_path = load_file_from_url(
                url=url, model_dir=path.join(model_path, resrgan_name), progress=True, file_name=None)

    # use ONNX acceleration, if available
    if params.platform == 'onnx':
        model = ONNXNet(ctx)
    else:
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                        num_block=23, num_grow_ch=32, scale=params.scale)

    dni_weight = None
    if resrgan_name == 'realesr-general-x4v3' and params.denoise != 1:
        wdn_model_path = model_path.replace(
            'realesr-general-x4v3', 'realesr-general-wdn-x4v3')
        model_path = [model_path, wdn_model_path]
        dni_weight = [params.denoise, 1 - params.denoise]

    upsampler = RealESRGANer(
        scale=params.scale,
        model_path=model_path,
        dni_weight=dni_weight,
        model=model,
        tile=tile,
        tile_pad=tile_pad,
        pre_pad=pre_pad,
        half=fp16)

    return upsampler


def upscale_resrgan(ctx: ServerContext, source_image: Image, params: UpscaleParams) -> Image:
    print('upscaling image with Real ESRGAN', params)

    image = np.array(source_image)
    upsampler = make_resrgan(ctx.model_path)

    # TODO: what is outscale for here?
    output, _ = upsampler.enhance(image, outscale=outscale)

    if params.faces:
        output = upscale_gfpgan(ctx, output)

    return Image.fromarray(output, 'RGB')


def upscale_gfpgan(ctx: ServerContext, image, upsampler=None) -> Image:
    print('correcting faces with GFPGAN')

    if upsampler is None:
        upsampler = make_resrgan(ctx.model_path, 512)

    face_enhancer = GFPGANer(
        model_path=gfpgan_url,
        upscale=outscale,
        arch='clean',
        channel_multiplier=2,
        bg_upsampler=upsampler)

    _, _, output = face_enhancer.enhance(
        image, has_aligned=False, only_center_face=False, paste_back=True)

    return output
