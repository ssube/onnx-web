from argparse import ArgumentParser
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from os import path, environ
from sys import exit
from torch.onnx import export

import torch

sources = {
    'diffusers': [
        # TODO: add stable diffusion models
    ],
    'gfpgan': [
        ('GFPGANv1.3', 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth'),
    ],
    'real_esrgan': [
        ('RealESRGAN_x4plus', 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'),
    ],
}

model_path = environ.get('ONNX_WEB_MODEL_PATH',
                         path.join('..', 'models'))


@torch.no_grad()
def convert_real_esrgan(name: str, url: str):
    dest_path = path.join(model_path, name)
    dest_onnx = path.join(model_path, name + '.onnx')
    print('converting Real ESRGAN into %s' % dest_path)

    if path.isfile(dest_onnx):
        print('Real ESRGAN ONNX model already exists, skipping.')
        return

    if not path.isfile(dest_path):
        print('PTH model not found, downloading...')
        dest_path = load_file_from_url(
            url=url, model_dir=path.join(dest_path, name), progress=True, file_name=None)

    print('loading and training Real ESRGAN model')
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                    num_block=23, num_grow_ch=32, scale=4)
    model.load_state_dict(torch.load(dest_path)['params_ema'])
    model.train(False)
    model.eval()

    rng = torch.rand(1, 3, 64, 64)
    input_names = ['data']
    output_names = ['output']
    dynamic_axes = {'data': {2: 'width', 3: 'height'},
                    'output': {2: 'width', 3: 'height'}}

    print('exporting Real ESRGAN model to %s' % dest_onnx)
    export(
        model,
        rng,
        dest_onnx,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=11,
        export_params=True
    )
    print('Real ESRGAN exported to ONNX')


def convert_gfpgan(name: str, url: str):
    dest_path = path.join(model_path, name)
    dest_onnx = path.join(model_path, name + '.onnx')

    print('converting GFPGAN into %s' % dest_path)

    if path.isfile(dest_onnx):
        print('GFPGAN ONNX model already exists, skipping.')
        return

    if not path.isfile(dest_path):
        print('existing model not found, downloading...')
        dest_path = load_file_from_url(
            url=url, model_dir=path.join(dest_path, name), progress=True, file_name=None)

    print('loading and training GFPGAN model')
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                    num_block=23, num_grow_ch=32, scale=4)

    # TODO: make sure strict=False is safe here
    model.load_state_dict(torch.load(dest_path)['params_ema'], strict=False)
    model.train(False)
    model.eval()

    rng = torch.rand(1, 3, 64, 64)
    input_names = ['data']
    output_names = ['output']
    dynamic_axes = {'data': {2: 'width', 3: 'height'},
                    'output': {2: 'width', 3: 'height'}}

    print('exporting GFPGAN model to %s' % dest_onnx)
    export(
        model,
        rng,
        dest_onnx,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=11,
        export_params=True
    )
    print('GFPGAN exported to ONNX')


def convert_diffuser():
    pass


def main() -> int:
    parser = ArgumentParser(
        prog='onnx-web model converter',
        description='convert checkpoint models to ONNX')
    parser.add_argument('--diffusers', type=str, nargs='*',
                        help='models using the diffusers pipeline')
    parser.add_argument('--gfpgan', action='store_true')
    parser.add_argument('--resrgan', action='store_true')
    args = parser.parse_args()
    print(args)

    if args.diffusers:
        for source in args.diffusers:
            print('converting Diffusers model: %s' % source[0])
            convert_diffuser(*source)

    if args.resrgan:
        for source in sources.get('real_esrgan'):
            print('converting Real ESRGAN model: %s' % source[0])
            convert_real_esrgan(*source)

    if args.gfpgan:
        for source in sources.get('gfpgan'):
            print('converting GFPGAN model: %s' % source[0])
            convert_gfpgan(*source)

    return 0


if __name__ == '__main__':
    exit(main())
