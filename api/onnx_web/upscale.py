from basicsr.archs.rrdbnet_arch import RRDBNet
from gfpgan import GFPGANer
from onnxruntime import InferenceSession
from os import path
from PIL import Image
from realesrgan import RealESRGANer
from typing import Any, Literal, Union

import numpy as np
import torch

from .utils import (
    ServerContext,
    Size,
)

# TODO: these should all be params or config
pre_pad = 0
tile_pad = 10


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

    def size(self):
        return np.shape(self.source)


class ONNXNet():
    '''
    Provides the RRDBNet interface using an ONNX session for DirectML acceleration.
    '''

    def __init__(self, ctx: ServerContext, model: str, provider='DmlExecutionProvider') -> None:
        '''
        TODO: get platform provider from request params
        '''
        model_path = path.join(ctx.model_path, model)
        self.session = InferenceSession(
            model_path, providers=[provider])

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
    def __init__(
        self,
        upscale_model: str,
        provider: str,
        correction_model: Union[str, None] = None,
        denoise: float = 0.5,
        faces=True,
        face_strength: float = 0.5,
        format: Literal['onnx', 'pth'] = 'onnx',
        half=False,
        outscale: int = 1,
        scale: int = 4,
    ) -> None:
        self.upscale_model = upscale_model
        self.provider = provider
        self.correction_model = correction_model
        self.denoise = denoise
        self.faces = faces
        self.face_strength = face_strength
        self.format = format
        self.half = half
        self.outscale = outscale
        self.scale = scale

    def resize(self, size: Size) -> Size:
        return Size(size.width * self.outscale, size.height * self.outscale)


def make_resrgan(ctx: ServerContext, params: UpscaleParams, tile=0):
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
    else:
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
        tile_pad=tile_pad,
        pre_pad=pre_pad,
        half=params.half)

    return upsampler


def upscale_resrgan(ctx: ServerContext, params: UpscaleParams, source_image: Image) -> Image:
    print('upscaling image with Real ESRGAN', params.scale)

    output = np.array(source_image)
    upsampler = make_resrgan(ctx, params, tile=512)

    if params.scale > 1:
        output, _ = upsampler.enhance(output, outscale=params.outscale)

    if params.faces:
        output = upscale_gfpgan(ctx, params, output, upsampler=upsampler)

    output = Image.fromarray(output, 'RGB')
    print('final output image size', output.size)
    return output


def upscale_gfpgan(ctx: ServerContext, params: UpscaleParams, image, upsampler=None) -> Image:
    print('correcting faces with GFPGAN model: %s' % params.correction_model)

    if params.correction_model is None:
        print('no face model given, skipping')
        return image

    if upsampler is None:
        upsampler = make_resrgan(ctx, params)

    face_path = path.join(ctx.model_path, '%s.pth' % (params.correction_model))

    # TODO: doesn't have a model param, not sure how to pass ONNX model
    face_enhancer = GFPGANer(
        model_path=face_path,
        upscale=params.outscale,
        arch='clean',
        channel_multiplier=2,
        bg_upsampler=upsampler)

    _, _, output = face_enhancer.enhance(
        image, has_aligned=False, only_center_face=False, paste_back=True, weight=params.face_strength)

    return output
