import warnings
from argparse import ArgumentParser
from logging import getLogger
from os import makedirs, path
from sys import exit
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

from jsonschema import ValidationError, validate
from yaml import safe_load

from .correction_gfpgan import convert_correction_gfpgan
from .diffusion_original import convert_diffusion_original
from .diffusion_stable import convert_diffusion_stable
from .upscale_resrgan import convert_upscale_resrgan
from .utils import (
    ConversionContext,
    download_progress,
    model_formats_original,
    source_format,
    tuple_to_correction,
    tuple_to_diffusion,
    tuple_to_upscaling,
)

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


model_sources: Dict[str, Tuple[str, str]] = {
    "civitai://": ("Civitai", "https://civitai.com/api/download/models/%s"),
}

model_source_huggingface = "huggingface://"

# recommended models
base_models: Models = {
    "diffusion": [
        # v1.x
        (
            "stable-diffusion-onnx-v1-5",
            model_source_huggingface + "runwayml/stable-diffusion-v1-5",
        ),
        (
            "stable-diffusion-onnx-v1-inpainting",
            model_source_huggingface + "runwayml/stable-diffusion-inpainting",
        ),
        # v2.x
        (
            "stable-diffusion-onnx-v2-1",
            model_source_huggingface + "stabilityai/stable-diffusion-2-1",
        ),
        (
            "stable-diffusion-onnx-v2-inpainting",
            model_source_huggingface + "stabilityai/stable-diffusion-2-inpainting",
        ),
        # TODO: should have its own converter
        (
            "upscaling-stable-diffusion-x4",
            model_source_huggingface + "stabilityai/stable-diffusion-x4-upscaler",
            True,
        ),
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


def fetch_model(
    ctx: ConversionContext, name: str, source: str, model_format: Optional[str] = None
) -> str:
    cache_name = path.join(ctx.cache_path, name)

    # add an extension if possible, some of the conversion code checks for it
    if model_format is None:
        url = urlparse(source)
        ext = path.basename(url.path)
        if ext is not None:
            cache_name = "%s.%s" % (cache_name, ext)
    else:
        cache_name = "%s.%s" % (cache_name, model_format)

    for proto in model_sources:
        api_name, api_root = model_sources.get(proto)
        if source.startswith(proto):
            api_source = api_root % (source.removeprefix(proto))
            logger.info(
                "Downloading model from %s: %s -> %s", api_name, api_source, cache_name
            )
            return download_progress([(api_source, cache_name)])

    if source.startswith(model_source_huggingface):
        hub_source = source.removeprefix(model_source_huggingface)
        logger.info("Downloading model from Huggingface Hub: %s", hub_source)
        # from_pretrained has a bunch of useful logic that snapshot_download by itself down not
        return hub_source
    elif source.startswith("https://"):
        logger.info("Downloading model from: %s", source)
        return download_progress([(source, cache_name)])
    elif source.startswith("http://"):
        logger.warning("Downloading model from insecure source: %s", source)
        return download_progress([(source, cache_name)])
    elif source.startswith(path.sep) or source.startswith("."):
        logger.info("Using local model: %s", source)
        return source
    else:
        logger.info("Unknown model location, using path as provided: %s", source)
        return source


def convert_models(ctx: ConversionContext, args, models: Models):
    if args.diffusion:
        for model in models.get("diffusion"):
            model = tuple_to_diffusion(model)
            name = model.get("name")

            if name in args.skip:
                logger.info("Skipping model: %s", name)
            else:
                model_format = source_format(model)
                source = fetch_model(
                    ctx, name, model["source"], model_format=model_format
                )

                if model_format in model_formats_original:
                    convert_diffusion_original(
                        ctx,
                        model,
                        source,
                    )
                else:
                    convert_diffusion_stable(
                        ctx,
                        model,
                        source,
                    )

    if args.upscaling:
        for model in models.get("upscaling"):
            model = tuple_to_upscaling(model)
            name = model.get("name")

            if name in args.skip:
                logger.info("Skipping model: %s", name)
            else:
                model_format = source_format(model)
                source = fetch_model(
                    ctx, name, model["source"], model_format=model_format
                )
                convert_upscale_resrgan(ctx, model, source)

    if args.correction:
        for model in models.get("correction"):
            model = tuple_to_correction(model)
            name = model.get("name")

            if name in args.skip:
                logger.info("Skipping model: %s", name)
            else:
                model_format = source_format(model)
                source = fetch_model(
                    ctx, name, model["source"], model_format=model_format
                )
                convert_correction_gfpgan(ctx, model, source)


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

    ctx = ConversionContext(half=args.half, opset=args.opset, token=args.token)
    logger.info("Converting models in %s using %s", ctx.model_path, ctx.training_device)

    if ctx.half and ctx.training_device != "cuda":
        raise ValueError(
            "Half precision model export is only supported on GPUs with CUDA"
        )

    if not path.exists(ctx.model_path):
        logger.info("Model path does not existing, creating: %s", ctx.model_path)
        makedirs(ctx.model_path)

    logger.info("Converting base models.")
    convert_models(ctx, args, base_models)

    for file in args.extras:
        if file is not None and file != "":
            logger.info("Loading extra models from %s", file)
            try:
                with open(file, "r") as f:
                    data = safe_load(f.read())

                with open("./schemas/extras.yaml", "r") as f:
                    schema = safe_load(f.read())

                logger.debug("validating chain request: %s against %s", data, schema)

                try:
                    validate(data, schema)
                    logger.info("Converting extra models.")
                    convert_models(ctx, args, data)
                except ValidationError as err:
                    logger.error("Invalid data in extras file: %s", err)
            except Exception as err:
                logger.error("Error converting extra models: %s", err)

    return 0


if __name__ == "__main__":
    exit(main())
