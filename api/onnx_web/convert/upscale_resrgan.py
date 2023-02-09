import torch
from shutil import copyfile
from basicsr.utils.download_util import load_file_from_url
from torch.onnx import export
from os import path
from logging import getLogger
from basicsr.archs.rrdbnet_arch import RRDBNet
from .utils import ConversionContext

logger = getLogger(__name__)


@torch.no_grad()
def convert_upscale_resrgan(ctx: ConversionContext, name: str, url: str, scale: int, opset: int):
    dest_path = path.join(ctx.model_path, name + ".pth")
    dest_onnx = path.join(ctx.model_path, name + ".onnx")
    logger.info("converting Real ESRGAN model: %s -> %s", name, dest_onnx)

    if path.isfile(dest_onnx):
        logger.info("ONNX model already exists, skipping.")
        return

    if not path.isfile(dest_path):
        logger.info("PTH model not found, downloading...")
        download_path = load_file_from_url(
            url=url, model_dir=dest_path + "-cache", progress=True, file_name=None
        )
        copyfile(download_path, dest_path)

    logger.info("loading and training model")
    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=scale,
    )

    torch_model = torch.load(dest_path, map_location=ctx.map_location)
    if "params_ema" in torch_model:
        model.load_state_dict(torch_model["params_ema"])
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

    logger.info("exporting ONNX model to %s", dest_onnx)
    export(
        model,
        rng,
        dest_onnx,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        export_params=True,
    )
    logger.info("Real ESRGAN exported to ONNX successfully.")
