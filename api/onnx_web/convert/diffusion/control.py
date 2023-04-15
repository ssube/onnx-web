from logging import getLogger
from os import path
from pathlib import Path
from typing import Dict

import torch

from ...constants import ONNX_MODEL
from ...diffusers.version_safe_diffusers import AttnProcessor, ControlNetModel
from ..utils import ConversionContext, is_torch_2_0, onnx_export

logger = getLogger(__name__)


@torch.no_grad()
def convert_diffusion_control(
    conversion: ConversionContext,
    model: Dict,
    source: str,
    model_path: str,
    output_path: str,
    opset: int,
    attention_slicing: str,
):
    name = model.get("name")
    source = source or model.get("source")

    device = conversion.training_device
    dtype = conversion.torch_dtype()
    logger.debug("using Torch dtype %s for ControlNet", dtype)

    output_path = Path(output_path)
    logger.info("converting ControlNet model %s: %s -> %s", name, source, output_path)
    if path.exists(output_path):
        logger.info("ONNX model already exists, skipping")
        return

    controlnet = ControlNetModel.from_pretrained(model_path, torch_dtype=dtype)
    if attention_slicing is not None:
        logger.info("enabling attention slicing for ControlNet")
        controlnet.set_attention_slice(attention_slicing)

    # UNET
    if is_torch_2_0:
        controlnet.set_attn_processor(AttnProcessor())

    cnet_path = output_path / "cnet" / ONNX_MODEL
    onnx_export(
        controlnet,
        model_args=(
            torch.randn(2, 4, 64, 64).to(device=device, dtype=dtype),
            torch.randn(2).to(device=device, dtype=dtype),
            torch.randn(2, 77, 768).to(device=device, dtype=dtype),
            torch.randn(2, 3, 512, 512).to(device=device, dtype=dtype),
        ),
        output_path=cnet_path,
        ordered_input_names=[
            "sample",
            "timestep",
            "encoder_hidden_states",
            "controlnet_cond",
            "return_dict",
        ],
        output_names=[
            "down_block_res_samples",
            "mid_block_res_sample",
        ],  # has to be different from "sample" for correct tracing
        dynamic_axes={
            "sample": {0: "batch", 1: "channels", 2: "height", 3: "width"},
            "timestep": {0: "batch"},
            "encoder_hidden_states": {0: "batch", 1: "sequence"},
            "controlnet_cond": {0: "batch", 2: "height", 3: "width"},
        },
        opset=opset,
    )

    logger.info("ONNX ControlNet saved to %s", output_path)
