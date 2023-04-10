from logging import getLogger
from os import path

import torch
from torch.onnx import export

from ...models.swinir import SwinIR
from ..utils import ConversionContext, ModelDict

logger = getLogger(__name__)


@torch.no_grad()
def convert_upscaling_swinir(
    conversion: ConversionContext,
    model: ModelDict,
    source: str,
):
    name = model.get("name")
    source = source or model.get("source")
    scale = model.get("scale", 1)

    dest = path.join(conversion.model_path, name + ".onnx")
    logger.info("converting SwinIR model: %s -> %s", name, dest)

    if path.isfile(dest):
        logger.info("ONNX model already exists, skipping")
        return

    logger.info("loading and training model")
    img_size = (64, 64)  # TODO: does this need to be a fixed value?
    model = SwinIR(
        depths=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        img_range=1.0,
        img_size=img_size,
        in_chans=3,
        mlp_ratio=2,
        num_heads=[6, 6, 6, 6, 6, 6],
        resi_connection="1conv",
        upscale=scale,
        upsampler="pixelshuffle",
        window_size=8,
    )

    torch_model = torch.load(source, map_location=conversion.map_location)
    if "params_ema" in torch_model:
        model.load_state_dict(torch_model["params_ema"], strict=False)
    else:
        model.load_state_dict(torch_model["params"], strict=False)

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
    logger.info("SwinIR exported to ONNX successfully")
