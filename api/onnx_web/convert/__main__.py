from .correction_gfpgan import convert_correction_gfpgan
from .diffusion_original import convert_diffusion_original
from .diffusion_stable import convert_diffusion_stable
from .upscale_resrgan import convert_upscale_resrgan
from .utils import ConversionContext

import warnings
from argparse import ArgumentParser
from json import loads
from logging import getLogger
from os import environ, makedirs, path
from sys import exit
from typing import Dict, List, Optional, Tuple

import torch

# suppress common but harmless warnings, https://github.com/ssube/onnx-web/issues/75
warnings.filterwarnings(
    "ignore", ".*The shape inference of prim::Constant type is missing.*"
)
warnings.filterwarnings("ignore", ".*Only steps=1 can be constant folded.*")
warnings.filterwarnings(
    "ignore",
    ".*Converting a tensor to a Python boolean might cause the trace to be incorrect.*",
)

Models = Dict[str, List[Tuple[str, str, Optional[int]]]]

logger = getLogger(__name__)


# recommended models
base_models: Models = {
    "diffusion": [
        # v1.x
        ("stable-diffusion-onnx-v1-5", "runwayml/stable-diffusion-v1-5"),
        ("stable-diffusion-onnx-v1-inpainting", "runwayml/stable-diffusion-inpainting"),
        # v2.x
        ("stable-diffusion-onnx-v2-1", "stabilityai/stable-diffusion-2-1"),
        (
            "stable-diffusion-onnx-v2-inpainting",
            "stabilityai/stable-diffusion-2-inpainting",
        ),
        # TODO: should have its own converter
        ("upscaling-stable-diffusion-x4", "stabilityai/stable-diffusion-x4-upscaler"),
        # TODO: testing safetensors
        ("diffusion-stably-diffused-onnx-v2-6", "../models/tensors/stablydiffuseds_26.safetensors"),
        ("diffusion-unstable-ink-dream-onnx-v6", "../models/tensors/unstableinkdream_v6.safetensors"),
    ],
    "correction": [
        (
            "correction-gfpgan-v1-3",
            "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth",
            4,
        ),
        (
            "correction-codeformer",
            "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
            1,
        ),
    ],
    "upscaling": [
        (
            "upscaling-real-esrgan-x2-plus",
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
            2,
        ),
        (
            "upscaling-real-esrgan-x4-plus",
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
            4,
        ),
        (
            "upscaling-real-esrgan-x4-v3",
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
            4,
        ),
    ],
}

model_path = environ.get("ONNX_WEB_MODEL_PATH", path.join("..", "models"))
training_device = "cuda" if torch.cuda.is_available() else "cpu"


def load_models(args, ctx: ConversionContext, models: Models):
    if args.diffusion:
        for source in models.get("diffusion"):
            name, file = source
            if name in args.skip:
                logger.info("Skipping model: %s", source[0])
            else:
                if file.endswith(".safetensors") or file.endswith(".ckpt"):
                    convert_diffusion_original(ctx, *source, args.opset, args.half)
                else:
                    # TODO: make this a parameter in the JSON/dict
                    single_vae = "upscaling" in source[0]
                    convert_diffusion_stable(
                        ctx, *source, args.opset, args.half, args.token, single_vae=single_vae
                    )

    if args.upscaling:
        for source in models.get("upscaling"):
            if source[0] in args.skip:
                logger.info("Skipping model: %s", source[0])
            else:
                convert_upscale_resrgan(ctx, *source, args.opset)

    if args.correction:
        for source in models.get("correction"):
            if source[0] in args.skip:
                logger.info("Skipping model: %s", source[0])
            else:
                convert_correction_gfpgan(ctx, *source, args.opset)


def main() -> int:
    parser = ArgumentParser(
        prog="onnx-web model converter", description="convert checkpoint models to ONNX"
    )

    # model groups
    parser.add_argument("--correction", action="store_true", default=False)
    parser.add_argument("--diffusion", action="store_true", default=False)
    parser.add_argument("--upscaling", action="store_true", default=False)

    # extra models
    parser.add_argument("--extras", nargs="*", type=str, default=[])
    parser.add_argument("--skip", nargs="*", type=str, default=[])

    # export options
    parser.add_argument(
        "--half",
        action="store_true",
        default=False,
        help="Export models for half precision, faster on some Nvidia cards.",
    )
    parser.add_argument(
        "--opset",
        default=14,
        type=int,
        help="The version of the ONNX operator set to use.",
    )
    parser.add_argument(
        "--token",
        type=str,
        help="HuggingFace token with read permissions for downloading models.",
    )

    args = parser.parse_args()
    logger.info("CLI arguments: %s", args)

    ctx = ConversionContext(model_path, training_device)
    logger.info("Converting models in %s using %s", ctx.model_path, ctx.training_device)

    if not path.exists(model_path):
        logger.info("Model path does not existing, creating: %s", model_path)
        makedirs(model_path)

    logger.info("Converting base models.")
    load_models(args, ctx, base_models)

    for file in args.extras:
        if file is not None and file != "":
            logger.info("Loading extra models from %s", file)
            try:
                with open(file, "r") as f:
                    data = loads(f.read())
                    logger.info("Converting extra models.")
                    load_models(args, ctx, data)
            except Exception as err:
                logger.error("Error converting extra models: %s", err)

    return 0


if __name__ == "__main__":
    exit(main())
