from argparse import ArgumentParser
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from os import path, environ
from sys import exit
from torch.onnx import export

import torch

from .upscale import (
    gfpgan_url,
    resrgan_url,
    resrgan_name,
)

model_path = environ.get('ONNX_WEB_MODEL_PATH',
                             path.join('..', 'models'))


@torch.no_grad()
def convert_real_esrgan():
    dest_path = path.join(model_path, resrgan_name + '.pth')
    print('converting Real ESRGAN into %s' % dest_path)

    if not path.isfile(dest_path):
        print('existing model not found, downloading...')
        for url in resrgan_url:
            dest_path = load_file_from_url(
                url=url, model_dir=path.join(dest_path, resrgan_name), progress=True, file_name=None)

    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                    num_block=23, num_grow_ch=32, scale=4)

    print('loading and training Real ESRGAN model')
    model.load_state_dict(torch.load(dest_path)['params_ema'])
    model.train(False)
    model.eval()

    rng = torch.rand(1, 3, 64, 64)
    input_names = ['data']
    output_names = ['output']
    dynamic_axes = {'data': {2: 'width', 3: 'height'},
                    'output': {2: 'width', 3: 'height'}}

    with torch.no_grad():
        dest_onnx = path.join(model_path, resrgan_name + '.onnx')
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


def convert_gfpgan():
    pass


def convert_diffuser():
    pass


def main() -> int:
    parser = ArgumentParser(
        prog='onnx-web model converter',
        description='convert checkpoint models to ONNX')
    parser.add_argument('--diffusers', type=str, nargs='+',
                        help='models using the diffusers pipeline')
    parser.add_argument('--gfpgan', action='store_true')
    parser.add_argument('--resrgan', action='store_true')
    args = parser.parse_args()
    print(args)

    for model in args.diffusers:
        print('convert ' + model)

    if args.resrgan:
        convert_real_esrgan()

    return 0


if __name__ == '__main__':
    exit(main())
