import warnings
from argparse import ArgumentParser
from logging import getLogger
from os import makedirs, path
from sys import exit
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from huggingface_hub.file_download import hf_hub_download
from jsonschema import ValidationError, validate
from onnx import load_model, save_model
from transformers import CLIPTokenizer
from yaml import safe_load

from onnx_web.convert.diffusion.control import convert_diffusion_control

from ..constants import ONNX_MODEL, ONNX_WEIGHTS
from .correction.gfpgan import convert_correction_gfpgan
from .diffusion.diffusers import convert_diffusion_diffusers
from .diffusion.lora import blend_loras
from .diffusion.original import convert_diffusion_original
from .diffusion.textual_inversion import blend_textual_inversions
from .upscaling.bsrgan import convert_upscaling_bsrgan
from .upscaling.resrgan import convert_upscale_resrgan
from .upscaling.swinir import convert_upscaling_swinir
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

Models = Dict[str, List[Any]]

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
        {
            "model": "swinir",
            "name": "upscaling-swinir-classical-x4",
            "source": "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth",
            "scale": 4,
        },
        {
            "model": "swinir",
            "name": "upscaling-swinir-real-large-x4",
            "source": "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth",
            "scale": 4,
        },
        {
            "model": "bsrgan",
            "name": "upscaling-bsrgan-x4",
            "source": "https://github.com/cszn/KAIR/releases/download/v1.0/BSRGAN.pth",
            "scale": 4,
        },
        {
            "model": "bsrgan",
            "name": "upscaling-bsrgan-x2",
            "source": "https://github.com/cszn/KAIR/releases/download/v1.0/BSRGANx2.pth",
            "scale": 2,
        },
    ],
    # download only
    "sources": [
        # CodeFormer: no ONNX yet
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
        # ControlNets: already converted
        {
            "dest": "control",
            "format": "onnx",
            "name": "canny",
            "source": "https://huggingface.co/ForserX/sd-controlnet-canny-onnx/resolve/main/model.onnx",
        },
        {
            "dest": "control",
            "format": "onnx",
            "name": "depth",
            "source": "https://huggingface.co/ForserX/sd-controlnet-depth-onnx/resolve/main/model.onnx",
        },
        {
            "dest": "control",
            "format": "onnx",
            "name": "hed",
            "source": "https://huggingface.co/ForserX/sd-controlnet-hed-onnx/resolve/main/model.onnx",
        },
        {
            "dest": "control",
            "format": "onnx",
            "name": "mlsd",
            "source": "https://huggingface.co/ForserX/sd-controlnet-mlsd-onnx/resolve/main/model.onnx",
        },
        {
            "dest": "control",
            "format": "onnx",
            "name": "normal",
            "source": "https://huggingface.co/ForserX/sd-controlnet-normal-onnx/resolve/main/model.onnx",
        },
        {
            "dest": "control",
            "format": "onnx",
            "name": "openpose",
            "source": "https://huggingface.co/ForserX/sd-controlnet-openpose-onnx/resolve/main/model.onnx",
        },
        {
            "dest": "control",
            "format": "onnx",
            "name": "seg",
            "source": "https://huggingface.co/ForserX/sd-controlnet-seg-onnx/resolve/main/model.onnx",
        },
    ],
}


def fetch_model(
    conversion: ConversionContext,
    name: str,
    source: str,
    dest: Optional[str] = None,
    format: Optional[str] = None,
    hf_hub_fetch: bool = False,
    hf_hub_filename: Optional[str] = None,
) -> str:
    cache_path = dest or conversion.cache_path
    cache_name = path.join(cache_path, name)

    # add an extension if possible, some of the conversion code checks for it
    if format is None:
        url = urlparse(source)
        ext = path.basename(url.path)
        _filename, ext = path.splitext(ext)
        if ext is not None:
            cache_name = cache_name + ext
    else:
        cache_name = f"{cache_name}.{format}"

    if path.exists(cache_name):
        logger.debug("model already exists in cache, skipping fetch")
        return cache_name

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
        if hf_hub_fetch:
            return hf_hub_download(
                repo_id=hub_source,
                filename=hf_hub_filename,
                cache_dir=cache_path,
                force_filename=f"{name}.bin",
            )
        else:
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


def convert_models(conversion: ConversionContext, args, models: Models):
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
                    dest_path = None
                    if "dest" in model:
                        dest_path = path.join(conversion.model_path, model["dest"])

                    dest = fetch_model(
                        conversion, name, source, format=model_format, dest=dest_path
                    )
                    logger.info("finished downloading source: %s -> %s", source, dest)
                except Exception:
                    logger.exception("error fetching source %s", name)

    if args.networks and "networks" in models:
        for network in models.get("networks"):
            name = network["name"]

            if name in args.skip:
                logger.info("skipping network: %s", name)
            else:
                network_format = source_format(network)
                network_model = network.get("model", None)
                network_type = network["type"]
                source = network["source"]

                try:
                    if network_type == "control":
                        dest = fetch_model(
                            conversion,
                            name,
                            source,
                            format=network_format,
                        )

                        convert_diffusion_control(
                            conversion,
                            network,
                            dest,
                        )
                    if network_type == "inversion" and network_model == "concept":
                        dest = fetch_model(
                            conversion,
                            name,
                            source,
                            dest=path.join(conversion.model_path, network_type),
                            format=network_format,
                            hf_hub_fetch=True,
                            hf_hub_filename="learned_embeds.bin",
                        )
                    else:
                        dest = fetch_model(
                            conversion,
                            name,
                            source,
                            dest=path.join(conversion.model_path, network_type),
                            format=network_format,
                        )

                    logger.info("finished downloading network: %s -> %s", source, dest)
                except Exception:
                    logger.exception("error fetching network %s", name)

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
                        conversion, name, model["source"], format=model_format
                    )

                    converted = False
                    if model_format in model_formats_original:
                        converted, dest = convert_diffusion_original(
                            conversion,
                            model,
                            source,
                        )
                    else:
                        converted, dest = convert_diffusion_diffusers(
                            conversion,
                            model,
                            source,
                        )

                    # make sure blending only happens once, not every run
                    if converted:
                        # keep track of which models have been blended
                        blend_models = {}

                        inversion_dest = path.join(conversion.model_path, "inversion")
                        lora_dest = path.join(conversion.model_path, "lora")

                        for inversion in model.get("inversions", []):
                            if "text_encoder" not in blend_models:
                                blend_models["text_encoder"] = load_model(
                                    path.join(
                                        dest,
                                        "text_encoder",
                                        ONNX_MODEL,
                                    )
                                )

                            if "tokenizer" not in blend_models:
                                blend_models[
                                    "tokenizer"
                                ] = CLIPTokenizer.from_pretrained(
                                    dest,
                                    subfolder="tokenizer",
                                )

                            inversion_name = inversion["name"]
                            inversion_source = inversion["source"]
                            inversion_format = inversion.get("format", None)
                            inversion_source = fetch_model(
                                conversion,
                                inversion_name,
                                inversion_source,
                                dest=inversion_dest,
                            )
                            inversion_token = inversion.get("token", inversion_name)
                            inversion_weight = inversion.get("weight", 1.0)

                            blend_textual_inversions(
                                conversion,
                                blend_models["text_encoder"],
                                blend_models["tokenizer"],
                                [
                                    (
                                        inversion_source,
                                        inversion_weight,
                                        inversion_token,
                                        inversion_format,
                                    )
                                ],
                            )

                        for lora in model.get("loras", []):
                            if "text_encoder" not in blend_models:
                                blend_models["text_encoder"] = load_model(
                                    path.join(
                                        dest,
                                        "text_encoder",
                                        ONNX_MODEL,
                                    )
                                )

                            if "unet" not in blend_models:
                                blend_models["unet"] = load_model(
                                    path.join(dest, "unet", ONNX_MODEL)
                                )

                            # load models if not loaded yet
                            lora_name = lora["name"]
                            lora_source = lora["source"]
                            lora_source = fetch_model(
                                conversion,
                                f"{name}-lora-{lora_name}",
                                lora_source,
                                dest=lora_dest,
                            )
                            lora_weight = lora.get("weight", 1.0)

                            blend_loras(
                                conversion,
                                blend_models["text_encoder"],
                                [(lora_source, lora_weight)],
                                "text_encoder",
                            )

                            blend_loras(
                                conversion,
                                blend_models["unet"],
                                [(lora_source, lora_weight)],
                                "unet",
                            )

                        if "tokenizer" in blend_models:
                            dest_path = path.join(dest, "tokenizer")
                            logger.debug("saving blended tokenizer to %s", dest_path)
                            blend_models["tokenizer"].save_pretrained(dest_path)

                        for name in ["text_encoder", "unet"]:
                            if name in blend_models:
                                dest_path = path.join(dest, name, ONNX_MODEL)
                                logger.debug(
                                    "saving blended %s model to %s", name, dest_path
                                )
                                save_model(
                                    blend_models[name],
                                    dest_path,
                                    save_as_external_data=True,
                                    all_tensors_to_one_file=True,
                                    location=ONNX_WEIGHTS,
                                )

                except Exception:
                    logger.exception(
                        "error converting diffusion model %s",
                        name,
                    )

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
                        conversion, name, model["source"], format=model_format
                    )
                    model_type = model.get("model", "resrgan")
                    if model_type == "bsrgan":
                        convert_upscaling_bsrgan(conversion, model, source)
                    elif model_type == "resrgan":
                        convert_upscale_resrgan(conversion, model, source)
                    elif model_type == "swinir":
                        convert_upscaling_swinir(conversion, model, source)
                    else:
                        logger.error(
                            "unknown upscaling model type %s for %s", model_type, name
                        )
                except Exception:
                    logger.exception(
                        "error converting upscaling model %s",
                        name,
                    )

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
                        conversion, name, model["source"], format=model_format
                    )
                    model_type = model.get("model", "gfpgan")
                    if model_type == "gfpgan":
                        convert_correction_gfpgan(conversion, model, source)
                    else:
                        logger.error(
                            "unknown correction model type %s for %s", model_type, name
                        )
                except Exception:
                    logger.exception(
                        "error converting correction model %s",
                        name,
                    )


def main() -> int:
    parser = ArgumentParser(
        prog="onnx-web model converter", description="convert checkpoint models to ONNX"
    )

    # model groups
    parser.add_argument("--networks", action="store_true", default=True)
    parser.add_argument("--sources", action="store_true", default=True)
    parser.add_argument("--correction", action="store_true", default=False)
    parser.add_argument("--diffusion", action="store_true", default=False)
    parser.add_argument("--upscaling", action="store_true", default=False)

    # extra models
    parser.add_argument("--extras", nargs="*", type=str, default=[])
    parser.add_argument("--prune", nargs="*", type=str, default=[])
    parser.add_argument("--skip", nargs="*", type=str, default=[])

    # export options
    parser.add_argument(
        "--half",
        action="store_true",
        default=False,
        help="Export models for half precision, smaller and faster on most GPUs.",
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

    server = ConversionContext.from_environ()
    server.half = args.half or "onnx-fp16" in server.optimizations
    server.opset = args.opset
    server.token = args.token
    logger.info(
        "converting models in %s using %s", server.model_path, server.training_device
    )

    if not path.exists(server.model_path):
        logger.info("model path does not existing, creating: %s", server.model_path)
        makedirs(server.model_path)

    logger.info("converting base models")
    convert_models(server, args, base_models)

    extras = []
    extras.extend(server.extra_models)
    extras.extend(args.extras)
    extras = list(set(extras))
    extras.sort()
    logger.debug("loading extra files: %s", extras)

    with open("./schemas/extras.yaml", "r") as f:
        extra_schema = safe_load(f.read())

    for file in extras:
        if file is not None and file != "":
            logger.info("loading extra models from %s", file)
            try:
                with open(file, "r") as f:
                    data = safe_load(f.read())

                logger.debug("validating extras file %s", data)
                try:
                    validate(data, extra_schema)
                    logger.info("converting extra models")
                    convert_models(server, args, data)
                except ValidationError:
                    logger.exception("invalid data in extras file")
            except Exception:
                logger.exception("error converting extra models")

    return 0


if __name__ == "__main__":
    exit(main())
