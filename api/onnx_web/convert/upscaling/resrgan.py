from logging import getLogger
from os import path

import torch
from torch.onnx import export

from ...models.rrdb import RRDBNet
from ..utils import ConversionContext, ModelDict

logger = getLogger(__name__)

TAG_X4_V3 = "real-esrgan-x4-v3"


@torch.no_grad()
def convert_upscale_resrgan(
    conversion: ConversionContext,
    model: ModelDict,
    source: str,
):
    from realesrgan.archs.srvgg_arch import SRVGGNetCompact

    name = model.get("name")
    source = source or model.get("source")
    scale = model.get("scale")

    dest = path.join(conversion.model_path, name + ".onnx")
    logger.info("converting Real ESRGAN model: %s -> %s", name, dest)

    if path.isfile(dest):
        logger.info("ONNX model already exists, skipping")
        return

    logger.info("loading and training model")

    if TAG_X4_V3 in name:
        # the x4-v3 model needs a different network
        model = SRVGGNetCompact(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_conv=32,
            upscale=scale,
            act_type="prelu",
        )
    else:
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
        model.load_state_dict(torch_model["params_ema"])
    else:
        model.load_state_dict(torch_model["params"], strict=False)

    model.to(conversion.training_device).train(False)
    model.eval()

    rng = torch.rand(1, 3, 64, 64, device=conversion.map_location)
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
        opset_version=conversion.opset,
        export_params=True,
    )
    logger.info("real ESRGAN exported to ONNX successfully")
