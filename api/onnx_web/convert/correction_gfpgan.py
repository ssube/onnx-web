from logging import getLogger
from os import path

import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from torch.onnx import export

from .utils import ConversionContext, ModelDict

logger = getLogger(__name__)


@torch.no_grad()
def convert_correction_gfpgan(
    ctx: ConversionContext,
    model: ModelDict,
    source: str,
):
    name = model.get("name")
    source = source or model.get("source")
    scale = model.get("scale")

    dest = path.join(ctx.model_path, name + ".onnx")
    logger.info("converting GFPGAN model: %s -> %s", name, dest)

    if path.isfile(dest):
        logger.info("ONNX model already exists, skipping.")
        return

    logger.info("loading and training model")
    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=scale,
    )

    torch_model = torch.load(source, map_location=ctx.map_location)
    # TODO: make sure strict=False is safe here
    if "params_ema" in torch_model:
        model.load_state_dict(torch_model["params_ema"], strict=False)
    else:
        model.load_state_dict(torch_model["params"], strict=False)

    model.to(ctx.training_device).train(False)
    model.eval()

    rng = torch.rand(1, 3, 64, 64, device=ctx.map_location)
    input_names = ["data"]
    output_names = ["output"]
    dynamic_axes = {
        "data": {2: "width", 3: "height"},
        "output": {2: "width", 3: "height"},
    }

    logger.info("exporting ONNX model to %s", dest)
    export(
        model,
        rng,
        dest,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=ctx.opset,
        export_params=True,
    )
    logger.info("GFPGAN exported to ONNX successfully.")
