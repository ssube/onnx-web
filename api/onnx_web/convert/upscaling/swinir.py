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
    # values based on https://github.com/JingyunLiang/SwinIR/blob/main/main_test_swinir.py#L128
    params = {
        "depths": [6] * 6,
        "embed_dim": 180,
        "img_range": 1.0,
        "img_size": (64, 64),
        "in_chans": 3,
        "num_heads": [6] * 6,
        "resi_connection": "1conv",
        "upsampler": "pixelshuffle",
        "window_size": 8,
    }

    if "lightweight" in name:
        logger.debug("using SwinIR lightweight params")
        params["depths"] = [6] * 4
        params["embed_dim"] = 60
        params["num_heads"] = [6] * 4
        params["upsampler"] = "pixelshuffledirect"
    elif "real" in name:
        if "large" in name:
            logger.debug("using SwinIR real large params")
            params["depths"] = [6] * 9
            params["embed_dim"] = 240
            params["num_heads"] = [8] * 9
            params["upsampler"] = "nearest+conv"
            params["resi_connection"] = "3conv"
        else:
            logger.debug("using SwinIR real params")
            params["upsampler"] = "nearest+conv"
    elif "gray_dn" in name:
        params["img_size"] = (128, 128)
        params["in_chans"] = 1
        params["upsampler"] = ""
    elif "color_dn" in name:
        params["img_size"] = (128, 128)
        params["upsampler"] = ""
    elif "gray_jpeg" in name:
        params["img_range"] = 255.0
        params["img_size"] = (126, 126)
        params["in_chans"] = 1
        params["upsampler"] = ""
        params["window_size"] = 7
    elif "color_jpeg" in name:
        params["img_range"] = 255.0
        params["img_size"] = (126, 126)
        params["upsampler"] = ""
        params["window_size"] = 7

    model = SwinIR(
        mlp_ratio=2,
        upscale=scale,
        **params,
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
    logger.info("SwinIR exported to ONNX successfully")
