from logging import getLogger
from os import path
from re import compile

import torch
from torch.onnx import export

from ...models.srvgg import SRVGGNetCompact
from ..utils import ConversionContext, ModelDict

logger = getLogger(__name__)

TAG_X4_V3 = "real-esrgan-x4-v3"

SPECIAL_KEYS = {
    "model.0.bias": "conv_first.bias",
    "model.0.weight": "conv_first.weight",
    "model.1.sub.23.bias": "conv_body.bias",
    "model.1.sub.23.weight": "conv_body.weight",
    "model.3.bias": "conv_up1.bias",
    "model.3.weight": "conv_up1.weight",
    "model.6.bias": "conv_up2.bias",
    "model.6.weight": "conv_up2.weight",
    "model.8.bias": "conv_hr.bias",
    "model.8.weight": "conv_hr.weight",
    "model.10.bias": "conv_last.bias",
    "model.10.weight": "conv_last.weight",
}

SUB_NAME = compile(r"^model\.1\.sub\.(\d+)\.RDB(\d)\.conv(\d)\.0\.(bias|weight)$")


def fix_resrgan_keys(model):
    original_keys = list(model.keys())
    for key in original_keys:
        if key in SPECIAL_KEYS:
            new_key = SPECIAL_KEYS[key]
        else:
            # convert RDBN keys
            matched = SUB_NAME.match(key)
            if matched is not None:
                sub_index, rdb_index, conv_index, node_type = matched.groups()
                new_key = (
                    f"body.{sub_index}.rdb{rdb_index}.conv{conv_index}.{node_type}"
                )
            else:
                raise ValueError("unknown key format")

        if new_key in model:
            raise ValueError("key collision")

        model[new_key] = model[key]
        del model[key]

    return model


@torch.no_grad()
def convert_upscale_resrgan(
    conversion: ConversionContext,
    model: ModelDict,
    source: str,
):
    from basicsr.archs.rrdbnet_arch import RRDBNet

    name = model.get("name")
    source = source or model.get("source")
    scale = model.get("scale")

    dest = path.join(conversion.model_path, name + ".onnx")
    logger.info("converting Real ESRGAN model: %s -> %s", name, dest)

    if path.isfile(dest):
        logger.info("ONNX model already exists, skipping")
        return

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
    elif "params" in torch_model:
        model.load_state_dict(torch_model["params"], strict=False)
    else:
        # keys need fixed up to match
        model.load_state_dict(fix_resrgan_keys(torch_model), strict=False)

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
