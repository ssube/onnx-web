from argparse import ArgumentParser
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from diffusers import OnnxRuntimeModel, OnnxStableDiffusionPipeline, StableDiffusionPipeline
from onnx import load, save_model
from os import mkdir, path, environ
from pathlib import Path
from shutil import copyfile, rmtree
from sys import exit
from torch.onnx import export
from typing import Dict, List, Tuple, Union

import torch
import warnings

# suppress common but harmless warnings, https://github.com/ssube/onnx-web/issues/75
warnings.filterwarnings('ignore', '.*The shape inference of prim::Constant type is missing.*')
warnings.filterwarnings('ignore', '.*Only steps=1 can be constant folded.*')
warnings.filterwarnings('ignore', '.*Converting a tensor to a Python boolean might cause the trace to be incorrect.*')

Models = Dict[str, List[Tuple[str, str, Union[int, None]]]]

# recommended models
base_models: Models = {
    'diffusion': [
        # v1.x
        ('stable-diffusion-onnx-v1-5', 'runwayml/stable-diffusion-v1-5'),
        ('stable-diffusion-onnx-v1-inpainting',
         'runwayml/stable-diffusion-inpainting'),
        # v2.x
        ('stable-diffusion-onnx-v2-1', 'stabilityai/stable-diffusion-2-1'),
        ('stable-diffusion-onnx-v2-inpainting',
         'stabilityai/stable-diffusion-2-inpainting'),
    ],
    'correction': [
        ('correction-gfpgan-v1-3',
         'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth', 4),
    ],
    'upscaling': [
        ('upscaling-real-esrgan-x2-plus',
         'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth', 2),
        ('upscaling-real-esrgan-x4-plus',
         'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth', 4),
        ('upscaling-real-esrgan-x4-v3',
         'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth', 4),
    ],
}

# other neat models
extra_models: Models = {
    'diffusion': [
        ('diffusion-knollingcase', 'Aybeeceedee/knollingcase'),
        ('diffusion-openjourney', 'prompthero/openjourney'),
    ],
    'correction': [],
    'upscaling': [],
}

model_path = environ.get('ONNX_WEB_MODEL_PATH',
                         path.join('..', 'models'))
training_device = 'cuda' if torch.cuda.is_available() else 'cpu'
map_location = torch.device(training_device)

@torch.no_grad()
def convert_real_esrgan(name: str, url: str, scale: int, opset: int):
    dest_path = path.join(model_path, name + '.pth')
    dest_onnx = path.join(model_path, name + '.onnx')
    print('converting Real ESRGAN model: %s -> %s' % (name, dest_onnx))

    if path.isfile(dest_onnx):
        print('ONNX model already exists, skipping.')
        return

    if not path.isfile(dest_path):
        print('PTH model not found, downloading...')
        download_path = load_file_from_url(
            url=url, model_dir=dest_path + '-cache', progress=True, file_name=None)
        copyfile(download_path, dest_path)

    print('loading and training model')
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                    num_block=23, num_grow_ch=32, scale=scale)

    torch_model = torch.load(dest_path, map_location=map_location)
    if 'params_ema' in torch_model:
        model.load_state_dict(torch_model['params_ema'])
    else:
        model.load_state_dict(torch_model['params'], strict=False)

    model.to(training_device).train(False)
    model.eval()

    rng = torch.rand(1, 3, 64, 64, device=map_location)
    input_names = ['data']
    output_names = ['output']
    dynamic_axes = {'data': {2: 'width', 3: 'height'},
                    'output': {2: 'width', 3: 'height'}}

    print('exporting ONNX model to %s' % dest_onnx)
    export(
        model,
        rng,
        dest_onnx,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        export_params=True
    )
    print('Real ESRGAN exported to ONNX successfully.')


@torch.no_grad()
def convert_gfpgan(name: str, url: str, scale: int, opset: int):
    dest_path = path.join(model_path, name + '.pth')
    dest_onnx = path.join(model_path, name + '.onnx')
    print('converting GFPGAN model: %s -> %s' % (name, dest_onnx))

    if path.isfile(dest_onnx):
        print('ONNX model already exists, skipping.')
        return

    if not path.isfile(dest_path):
        print('PTH model not found, downloading...')
        download_path = load_file_from_url(
            url=url, model_dir=dest_path + '-cache', progress=True, file_name=None)
        copyfile(download_path, dest_path)

    print('loading and training model')
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                    num_block=23, num_grow_ch=32, scale=scale)

    torch_model = torch.load(dest_path, map_location=map_location)
    # TODO: make sure strict=False is safe here
    if 'params_ema' in torch_model:
        model.load_state_dict(torch_model['params_ema'], strict=False)
    else:
        model.load_state_dict(torch_model['params'], strict=False)

    model.to(training_device).train(False)
    model.eval()

    rng = torch.rand(1, 3, 64, 64, device=map_location)
    input_names = ['data']
    output_names = ['output']
    dynamic_axes = {'data': {2: 'width', 3: 'height'},
                    'output': {2: 'width', 3: 'height'}}

    print('exporting ONNX model to %s' % dest_onnx)
    export(
        model,
        rng,
        dest_onnx,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        export_params=True
    )
    print('GFPGAN exported to ONNX successfully.')


def onnx_export(
    model,
    model_args: tuple,
    output_path: Path,
    ordered_input_names,
    output_names,
    dynamic_axes,
    opset,
    use_external_data_format=False,
):
    '''
    From https://github.com/huggingface/diffusers/blob/main/scripts/convert_stable_diffusion_checkpoint_to_onnx.py
    '''

    output_path.parent.mkdir(parents=True, exist_ok=True)
    export(
        model,
        model_args,
        f=output_path.as_posix(),
        input_names=ordered_input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        opset_version=opset,
    )


@torch.no_grad()
def convert_diffuser(name: str, url: str, opset: int, half: bool, token: str):
    '''
    From https://github.com/huggingface/diffusers/blob/main/scripts/convert_stable_diffusion_checkpoint_to_onnx.py
    '''
    dtype = torch.float16 if half else torch.float32
    dest_path = path.join(model_path, name)

    # diffusers go into a directory rather than .onnx file
    print('converting Diffusers model: %s -> %s/' % (name, dest_path))

    if path.isdir(dest_path):
        print('ONNX model already exists, skipping.')
        return

    if half and training_device != 'cuda':
        raise ValueError(
            'Half precision model export is only supported on GPUs with CUDA')

    pipeline = StableDiffusionPipeline.from_pretrained(
        url, torch_dtype=dtype, use_auth_token=token).to(training_device)
    output_path = Path(dest_path)

    # TEXT ENCODER
    num_tokens = pipeline.text_encoder.config.max_position_embeddings
    text_hidden_size = pipeline.text_encoder.config.hidden_size
    text_input = pipeline.tokenizer(
        "A sample prompt",
        padding="max_length",
        max_length=pipeline.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    onnx_export(
        pipeline.text_encoder,
        # casting to torch.int32 until the CLIP fix is released: https://github.com/huggingface/transformers/pull/18515/files
        model_args=(text_input.input_ids.to(
            device=training_device, dtype=torch.int32)),
        output_path=output_path / "text_encoder" / "model.onnx",
        ordered_input_names=["input_ids"],
        output_names=["last_hidden_state", "pooler_output"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
        },
        opset=opset,
    )
    del pipeline.text_encoder

    # UNET
    unet_in_channels = pipeline.unet.config.in_channels
    unet_sample_size = pipeline.unet.config.sample_size
    unet_path = output_path / "unet" / "model.onnx"
    onnx_export(
        pipeline.unet,
        model_args=(
            torch.randn(2, unet_in_channels, unet_sample_size, unet_sample_size).to(
                device=training_device, dtype=dtype),
            torch.randn(2).to(device=training_device, dtype=dtype),
            torch.randn(2, num_tokens, text_hidden_size).to(
                device=training_device, dtype=dtype),
            False,
        ),
        output_path=unet_path,
        ordered_input_names=["sample", "timestep",
                             "encoder_hidden_states", "return_dict"],
        # has to be different from "sample" for correct tracing
        output_names=["out_sample"],
        dynamic_axes={
            "sample": {0: "batch", 1: "channels", 2: "height", 3: "width"},
            "timestep": {0: "batch"},
            "encoder_hidden_states": {0: "batch", 1: "sequence"},
        },
        opset=opset,
        use_external_data_format=True,  # UNet is > 2GB, so the weights need to be split
    )
    unet_model_path = str(unet_path.absolute().as_posix())
    unet_dir = path.dirname(unet_model_path)
    unet = load(unet_model_path)
    # clean up existing tensor files
    rmtree(unet_dir)
    mkdir(unet_dir)
    # collate external tensor files into one
    save_model(
        unet,
        unet_model_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="weights.pb",
        convert_attribute=False,
    )
    del pipeline.unet

    # VAE ENCODER
    vae_encoder = pipeline.vae
    vae_in_channels = vae_encoder.config.in_channels
    vae_sample_size = vae_encoder.config.sample_size
    # need to get the raw tensor output (sample) from the encoder
    vae_encoder.forward = lambda sample, return_dict: vae_encoder.encode(
        sample, return_dict)[0].sample()
    onnx_export(
        vae_encoder,
        model_args=(
            torch.randn(1, vae_in_channels, vae_sample_size, vae_sample_size).to(
                device=training_device, dtype=dtype),
            False,
        ),
        output_path=output_path / "vae_encoder" / "model.onnx",
        ordered_input_names=["sample", "return_dict"],
        output_names=["latent_sample"],
        dynamic_axes={
            "sample": {0: "batch", 1: "channels", 2: "height", 3: "width"},
        },
        opset=opset,
    )

    # VAE DECODER
    vae_decoder = pipeline.vae
    vae_latent_channels = vae_decoder.config.latent_channels
    vae_out_channels = vae_decoder.config.out_channels
    # forward only through the decoder part
    vae_decoder.forward = vae_encoder.decode
    onnx_export(
        vae_decoder,
        model_args=(
            torch.randn(1, vae_latent_channels, unet_sample_size, unet_sample_size).to(
                device=training_device, dtype=dtype),
            False,
        ),
        output_path=output_path / "vae_decoder" / "model.onnx",
        ordered_input_names=["latent_sample", "return_dict"],
        output_names=["sample"],
        dynamic_axes={
            "latent_sample": {0: "batch", 1: "channels", 2: "height", 3: "width"},
        },
        opset=opset,
    )
    del pipeline.vae

    # SAFETY CHECKER
    if pipeline.safety_checker is not None:
        safety_checker = pipeline.safety_checker
        clip_num_channels = safety_checker.config.vision_config.num_channels
        clip_image_size = safety_checker.config.vision_config.image_size
        safety_checker.forward = safety_checker.forward_onnx
        onnx_export(
            pipeline.safety_checker,
            model_args=(
                torch.randn(
                    1,
                    clip_num_channels,
                    clip_image_size,
                    clip_image_size,
                ).to(device=training_device, dtype=dtype),
                torch.randn(1, vae_sample_size, vae_sample_size, vae_out_channels).to(
                    device=training_device, dtype=dtype),
            ),
            output_path=output_path / "safety_checker" / "model.onnx",
            ordered_input_names=["clip_input", "images"],
            output_names=["out_images", "has_nsfw_concepts"],
            dynamic_axes={
                "clip_input": {0: "batch", 1: "channels", 2: "height", 3: "width"},
                "images": {0: "batch", 1: "height", 2: "width", 3: "channels"},
            },
            opset=opset,
        )
        del pipeline.safety_checker
        safety_checker = OnnxRuntimeModel.from_pretrained(
            output_path / "safety_checker")
        feature_extractor = pipeline.feature_extractor
    else:
        safety_checker = None
        feature_extractor = None

    onnx_pipeline = OnnxStableDiffusionPipeline(
        vae_encoder=OnnxRuntimeModel.from_pretrained(
            output_path / "vae_encoder"),
        vae_decoder=OnnxRuntimeModel.from_pretrained(
            output_path / "vae_decoder"),
        text_encoder=OnnxRuntimeModel.from_pretrained(
            output_path / "text_encoder"),
        tokenizer=pipeline.tokenizer,
        unet=OnnxRuntimeModel.from_pretrained(output_path / "unet"),
        scheduler=pipeline.scheduler,
        safety_checker=safety_checker,
        feature_extractor=feature_extractor,
        requires_safety_checker=safety_checker is not None,
    )

    print('exporting ONNX model')

    onnx_pipeline.save_pretrained(output_path)
    print("ONNX pipeline saved to", output_path)

    del pipeline
    del onnx_pipeline
    _ = OnnxStableDiffusionPipeline.from_pretrained(
        output_path, provider="CPUExecutionProvider")
    print("ONNX pipeline is loadable")


def load_models(args, models: Models):
    if args.diffusion:
        for source in models.get('diffusion'):
            if source[0] in args.skip:
                print('Skipping model: %s' % source[0])
            else:
                convert_diffuser(*source, args.opset, args.half, args.token)

    if args.upscaling:
        for source in models.get('upscaling'):
            if source[0] in args.skip:
                print('Skipping model: %s' % source[0])
            else:
                convert_real_esrgan(*source, args.opset)

    if args.correction:
        for source in models.get('correction'):
            if source[0] in args.skip:
                print('Skipping model: %s' % source[0])
            else:
                convert_gfpgan(*source, args.opset)


def main() -> int:
    parser = ArgumentParser(
        prog='onnx-web model converter',
        description='convert checkpoint models to ONNX')

    # model groups
    parser.add_argument('--correction', action='store_true', default=False)
    parser.add_argument('--diffusion', action='store_true', default=False)
    parser.add_argument('--extras', action='store_true', default=False)
    parser.add_argument('--upscaling', action='store_true', default=False)
    parser.add_argument('--skip', nargs='*', type=str, default=[])

    # export options
    parser.add_argument(
        '--half',
        action='store_true',
        default=False,
        help='Export models for half precision, faster on some Nvidia cards'
    )
    parser.add_argument(
        '--opset',
        default=14,
        type=int,
        help="The version of the ONNX operator set to use.",
    )
    parser.add_argument(
        '--token',
        type=str,
        help="HuggingFace token with read permissions for downloading models.",
    )

    args = parser.parse_args()
    print(args)

    print('Converting base models.')
    load_models(args, base_models)

    if args.extras:
        print('Converting extra models.')
        load_models(args, extra_models)

    return 0


if __name__ == '__main__':
    exit(main())
