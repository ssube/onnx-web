from logging import getLogger
from os import path
from typing import Dict, Optional, Tuple

import torch
from diffusers import StableDiffusionXLPipeline
from optimum.exporters.onnx import main_export

from ..utils import ConversionContext

logger = getLogger(__name__)


@torch.no_grad()
def convert_diffusion_diffusers_xl(
    conversion: ConversionContext,
    model: Dict,
    source: str,
    format: Optional[str],
    hf: bool = False,
) -> Tuple[bool, str]:
    """
    From https://github.com/huggingface/diffusers/blob/main/scripts/convert_stable_diffusion_checkpoint_to_onnx.py
    """
    name = model.get("name")
    # TODO: support alternate VAE

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

    if "hash" in model and not path.exists(model_hash):
        logger.info("ONNX model does not have hash file, adding one")
        with open(model_hash, "w") as f:
            f.write(model["hash"])

    if path.exists(dest_path) and path.exists(model_index):
        logger.info("ONNX model already exists, skipping conversion")
        return (False, dest_path)

    # safetensors -> diffusers directory with torch models
    temp_path = path.join(conversion.cache_path, f"{name}-torch")

    if format == "safetensors":
        pipeline = StableDiffusionXLPipeline.from_single_file(
            source, use_safetensors=True
        )
    else:
        pipeline = StableDiffusionXLPipeline.from_pretrained(source)

    pipeline.save_pretrained(temp_path)

    # directory -> onnx using optimum exporters
    main_export(
        temp_path,
        output=dest_path,
        task="stable-diffusion-xl",
        device=device,
        fp16=conversion.half,
        framework="pt",
    )

    # TODO: optimize UNet to fp16

    return False, dest_path
