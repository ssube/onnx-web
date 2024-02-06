from logging import getLogger
from os import path
from typing import Dict, Optional, Tuple

import onnx
import torch
from diffusers import AutoencoderKL, StableDiffusionXLPipeline
from onnx.shape_inference import infer_shapes_path
from onnxruntime.transformers.float16 import convert_float_to_float16
from optimum.exporters.onnx import main_export

from ...constants import ONNX_MODEL, ONNX_WEIGHTS
from ...utils import run_gc
from ..client import fetch_model
from ..utils import RESOLVE_FORMATS, ConversionContext, check_ext

logger = getLogger(__name__)


@torch.no_grad()
def convert_diffusion_diffusers_xl(
    conversion: ConversionContext,
    model: Dict,
    format: Optional[str],
) -> Tuple[bool, str]:
    """
    From https://github.com/huggingface/diffusers/blob/main/scripts/convert_stable_diffusion_checkpoint_to_onnx.py
    """
    name = str(model.get("name")).strip()
    source = model.get("source")
    replace_vae = model.get("vae", None)

    device = conversion.training_device
    dtype = conversion.torch_dtype()
    logger.debug("using Torch dtype %s for pipeline", dtype)

    dest_path = path.join(conversion.model_path, name)
    model_index = path.join(dest_path, "model_index.json")
    model_hash = path.join(dest_path, "hash.txt")

    # diffusers go into a directory rather than .onnx file
    logger.info(
        "converting Stable Diffusion XL model %s: %s -> %s/", name, source, dest_path
    )

    if path.exists(dest_path) and path.exists(model_index):
        logger.info("ONNX model already exists, skipping conversion")

        if "hash" in model and not path.exists(model_hash):
            logger.info("ONNX model does not have hash file, adding one")
            with open(model_hash, "w") as f:
                f.write(model["hash"])

        return (False, dest_path)

    cache_path = fetch_model(conversion, name, model["source"], format=format)
    # safetensors -> diffusers directory with torch models
    temp_path = path.join(conversion.cache_path, f"{name}-torch")

    if format == "safetensors":
        pipeline = StableDiffusionXLPipeline.from_single_file(
            cache_path, use_safetensors=True
        )
    else:
        pipeline = StableDiffusionXLPipeline.from_pretrained(cache_path)

    if replace_vae is not None:
        vae_path = path.join(conversion.model_path, replace_vae)
        vae_file = check_ext(vae_path, RESOLVE_FORMATS)
        if vae_file[0]:
            logger.debug("loading VAE from single tensor file: %s", vae_path)
            pipeline.vae = AutoencoderKL.from_single_file(vae_path)
        else:
            logger.debug("loading pretrained VAE from path: %s", replace_vae)
            pipeline.vae = AutoencoderKL.from_pretrained(replace_vae)

    if path.exists(temp_path):
        logger.debug("torch model already exists for %s: %s", source, temp_path)
    else:
        logger.debug("exporting torch model for %s: %s", source, temp_path)
        pipeline.save_pretrained(temp_path)

    # GC temporary pipeline
    del pipeline
    run_gc()

    # directory -> onnx using optimum exporters
    main_export(
        temp_path,
        output=dest_path,
        task="stable-diffusion-xl",
        device=device,
        fp16=conversion.has_optimization(
            "torch-fp16"
        ),  # optimum's fp16 mode only works on CUDA or ROCm
        framework="pt",
        do_validation=conversion.validate,
    )

    if "hash" in model:
        logger.debug("adding hash file to ONNX model")
        with open(model_hash, "w") as f:
            f.write(model["hash"])

    if conversion.half:
        unet_path = path.join(dest_path, "unet", ONNX_MODEL)
        infer_shapes_path(unet_path)
        unet = onnx.load(unet_path)
        opt_model = convert_float_to_float16(
            unet,
            disable_shape_infer=True,
            force_fp16_initializers=True,
            keep_io_types=True,
        )
        onnx.save_model(
            opt_model,
            unet_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=ONNX_WEIGHTS,
        )

    return (True, dest_path)
