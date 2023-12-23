from logging import getLogger
from os import path

import torch
from torch.onnx import export

from ...models.rrdb import RRDBNet
from ..utils import ConversionContext, ModelDict

logger = getLogger(__name__)


@torch.no_grad()
def convert_upscaling_bsrgan(
    conversion: ConversionContext,
    model: ModelDict,
    source: str,
):
    name = model.get("name")
    source = source or model.get("source")
    scale = model.get("scale", 1)

    dest = path.join(conversion.model_path, name + ".onnx")
    logger.info("converting BSRGAN model: %s -> %s", name, dest)

    if path.isfile(dest):
        logger.info("ONNX model already exists, skipping")
        return

    # values based on https://github.com/cszn/BSRGAN/blob/main/main_test_bsrgan.py#L69
    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=scale,
    )

    torch_model = torch.load(source, map_location=conversion.map_location)
    if "params_ema" in torch_model:
        model.load_state_dict(torch_model["params_ema"], strict=False)
    elif "params" in torch_model:
        model.load_state_dict(torch_model["params"], strict=False)
    else:
        model.load_state_dict(torch_model, strict=False)

    model.to(conversion.training_device).train(False)
    model.eval()

    rng = torch.rand(1, 3, 64, 64, device=conversion.map_location)
    input_names = ["input"]
    output_names = ["output"]
    dynamic_axes = {
        "input": {2: "h", 3: "w"},
        "output": {2: "h", 3: "w"},
    }

    logger.info("exporting ONNX model to %s", dest)
    export(
        model,
        rng,
        dest,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=conversion.opset,
        export_params=True,
    )
    logger.info("BSRGAN exported to ONNX successfully")
