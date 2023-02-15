from logging import getLogger
from os import path

import numpy as np
from PIL import Image

from ..onnx import OnnxNet
from ..params import DeviceParams, ImageParams, StageParams, UpscaleParams
from ..server.device_pool import JobContext
from ..utils import ServerContext, run_gc

logger = getLogger(__name__)

last_pipeline_instance = None
last_pipeline_params = (None, None)

x4_v3_tag = "real-esrgan-x4-v3"


def load_resrgan(
    server: ServerContext, params: UpscaleParams, device: DeviceParams, tile=0
):
    # must be within load function for patches to take effect
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
    from realesrgan.archs.srvgg_arch import SRVGGNetCompact

    model_file = "%s.%s" % (params.upscale_model, params.format)
    model_path = path.join(server.model_path, model_file)

    cache_key = (model_path, params.format)
    cache_pipe = server.cache.get("resrgan", cache_key)
    if cache_pipe is not None:
        logger.info("reusing existing Real ESRGAN pipeline")
        return cache_pipe

    if not path.isfile(model_path):
        raise Exception("Real ESRGAN model not found at %s" % model_path)

    if x4_v3_tag in model_file:
        # the x4-v3 model needs a different network
        model = SRVGGNetCompact(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_conv=32,
            upscale=4,
            act_type="prelu",
        )
    elif params.format == "onnx":
        # use ONNX acceleration, if available
        model = OnnxNet(
            server,
            model_file,
            provider=device.ort_provider(),
            sess_options=device.sess_options(),
        )
    elif params.format == "pth":
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=params.scale,
        )
    else:
        raise Exception("unknown platform %s" % params.format)

    dni_weight = None
    if params.upscale_model == x4_v3_tag and params.denoise != 1:
        wdn_model_path = model_path.replace(x4_v3_tag, "%s-wdn" % (x4_v3_tag))
        model_path = [model_path, wdn_model_path]
        dni_weight = [params.denoise, 1 - params.denoise]

    logger.debug("loading Real ESRGAN upscale model from %s", model_path)

    # TODO: shouldn't need the PTH file
    model_path_pth = path.join(server.cache_path, ("%s.pth" % params.upscale_model))
    upsampler = RealESRGANer(
        scale=params.scale,
        model_path=model_path_pth,
        dni_weight=dni_weight,
        model=model,
        tile=tile,
        tile_pad=params.tile_pad,
        pre_pad=params.pre_pad,
        half=params.half,
    )

    server.cache.set("resrgan", cache_key, upsampler)
    run_gc()

    return upsampler


def upscale_resrgan(
    job: JobContext,
    server: ServerContext,
    stage: StageParams,
    _params: ImageParams,
    source_image: Image.Image,
    *,
    upscale: UpscaleParams,
    **kwargs,
) -> Image.Image:
    logger.info("upscaling image with Real ESRGAN: x%s", upscale.scale)

    output = np.array(source_image)
    upsampler = load_resrgan(server, upscale, job.get_device(), tile=stage.tile_size)

    output, _ = upsampler.enhance(output, outscale=upscale.outscale)

    output = Image.fromarray(output, "RGB")
    logger.info("final output image size: %sx%s", output.width, output.height)
    return output
