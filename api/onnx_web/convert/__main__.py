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
from .diffusion.original import convert_diffusion_original
from .diffusion.diffusers import convert_diffusion_diffusers
from .diffusion.textual_inversion import convert_diffusion_textual_inversion
from .upscale_resrgan import convert_upscale_resrgan
from .utils import (
    ConversionContext,
    download_progress,
    model_formats_original,
    remove_prefix,
    source_format,
    tuple_to_correction,
    tuple_to_diffusion,
    tuple_to_source,
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
    # download only
    "sources": [
        (
            "detection-resnet50-final",
            "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/detection_Resnet50_Final.pth",
        ),
        (
            "detection-mobilenet025-final",
            "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/detection_mobilenet0.25_Final.pth",
        ),
        (
            "detection-yolo-v5-l",
            "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/yolov5l-face.pth",
        ),
        (
            "detection-yolo-v5-n",
            "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/yolov5n-face.pth",
        ),
        (
            "parsing-bisenet",
            "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/parsing_bisenet.pth",
        ),
        (
            "parsing-parsenet",
            "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/parsing_parsenet.pth",
        ),
    ],
}


def fetch_model(
    ctx: ConversionContext, name: str, source: str, model_format: Optional[str] = None
) -> str:
    cache_name = path.join(ctx.cache_path, name)
    model_path = path.join(ctx.model_path, name)
    model_onnx = model_path + ".onnx"

    for p in [model_path, model_onnx]:
        if path.exists(p):
            logger.debug("model already exists, skipping fetch")
            return p

    # add an extension if possible, some of the conversion code checks for it
    if model_format is None:
        url = urlparse(source)
        ext = path.basename(url.path)
        _filename, ext = path.splitext(ext)
        if ext is not None:
            cache_name += ext
    else:
        cache_name = "%s.%s" % (cache_name, model_format)

    for proto in model_sources:
        api_name, api_root = model_sources.get(proto)
        if source.startswith(proto):
            api_source = api_root % (remove_prefix(source, proto))
            logger.info(
                "downloading model from %s: %s -> %s", api_name, api_source, cache_name
            )
            return download_progress([(api_source, cache_name)])

    if source.startswith(model_source_huggingface):
        hub_source = remove_prefix(source, model_source_huggingface)
        logger.info("downloading model from Huggingface Hub: %s", hub_source)
        # from_pretrained has a bunch of useful logic that snapshot_download by itself down not
        return hub_source
    elif source.startswith("https://"):
        logger.info("downloading model from: %s", source)
        return download_progress([(source, cache_name)])
    elif source.startswith("http://"):
        logger.warning("downloading model from insecure source: %s", source)
        return download_progress([(source, cache_name)])
    elif source.startswith(path.sep) or source.startswith("."):
        logger.info("using local model: %s", source)
        return source
    else:
        logger.info("unknown model location, using path as provided: %s", source)
        return source


def convert_models(ctx: ConversionContext, args, models: Models):
    if args.sources and "sources" in models:
        for model in models.get("sources"):
            model = tuple_to_source(model)
            name = model.get("name")

            if name in args.skip:
                logger.info("skipping source: %s", name)
            else:
                model_format = source_format(model)
                source = model["source"]

                try:
                    dest = fetch_model(ctx, name, source, model_format=model_format)
                    logger.info("finished downloading source: %s -> %s", source, dest)
                except Exception as e:
                    logger.error("error fetching source %s: %s", name, e)

    if args.diffusion and "diffusion" in models:
        for model in models.get("diffusion"):
            model = tuple_to_diffusion(model)
            name = model.get("name")

            if name in args.skip:
                logger.info("skipping model: %s", name)
            else:
                model_format = source_format(model)

                try:
                    source = fetch_model(
                        ctx, name, model["source"], model_format=model_format
                    )

                    if "inversion" in model:
                        convert_diffusion_textual_inversion(ctx, source, model["inversion"])

                    if model_format in model_formats_original:
                        convert_diffusion_original(
                            ctx,
                            model,
                            source,
                        )
                    else:
                        convert_diffusion_diffusers(
                            ctx,
                            model,
                            source,
                        )
                except Exception as e:
                    logger.error("error converting diffusion model %s: %s", name, e)

    if args.upscaling and "upscaling" in models:
        for model in models.get("upscaling"):
            model = tuple_to_upscaling(model)
            name = model.get("name")

            if name in args.skip:
                logger.info("skipping model: %s", name)
            else:
                model_format = source_format(model)

                try:
                    source = fetch_model(
                        ctx, name, model["source"], model_format=model_format
                    )
                    convert_upscale_resrgan(ctx, model, source)
                except Exception as e:
                    logger.error("error converting upscaling model %s: %s", name, e)

    if args.correction and "correction" in models:
        for model in models.get("correction"):
            model = tuple_to_correction(model)
            name = model.get("name")

            if name in args.skip:
                logger.info("skipping model: %s", name)
            else:
                model_format = source_format(model)
                try:
                    source = fetch_model(
                        ctx, name, model["source"], model_format=model_format
                    )
                    convert_correction_gfpgan(ctx, model, source)
                except Exception as e:
                    logger.error("error converting correction model %s: %s", name, e)


def main() -> int:
    parser = ArgumentParser(
        prog="onnx-web model converter", description="convert checkpoint models to ONNX"
    )

    # model groups
    parser.add_argument("--sources", action="store_true", default=False)
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

    ctx = ConversionContext.from_environ()
    ctx.half = args.half
    ctx.opset = args.opset
    ctx.token = args.token
    logger.info("converting models in %s using %s", ctx.model_path, ctx.training_device)

    if ctx.half and ctx.training_device != "cuda":
        raise ValueError(
            "half precision model export is only supported on GPUs with CUDA"
        )

    if not path.exists(ctx.model_path):
        logger.info("model path does not existing, creating: %s", ctx.model_path)
        makedirs(ctx.model_path)

    logger.info("converting base models")
    convert_models(ctx, args, base_models)

    for file in args.extras:
        if file is not None and file != "":
            logger.info("loading extra models from %s", file)
            try:
                with open(file, "r") as f:
                    data = safe_load(f.read())

                with open("./schemas/extras.yaml", "r") as f:
                    schema = safe_load(f.read())

                logger.debug("validating chain request: %s against %s", data, schema)

                try:
                    validate(data, schema)
                    logger.info("converting extra models")
                    convert_models(ctx, args, data)
                except ValidationError as err:
                    logger.error("invalid data in extras file: %s", err)
            except Exception as err:
                logger.error("error converting extra models: %s", err)

    return 0


if __name__ == "__main__":
    exit(main())
