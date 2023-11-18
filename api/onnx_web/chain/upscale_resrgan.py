from logging import getLogger
from os import path
from typing import List, Optional

import numpy as np
from PIL import Image

from ..onnx import OnnxRRDBNet
from ..params import DeviceParams, ImageParams, StageParams, UpscaleParams
from ..server import ModelTypes, ServerContext
from ..utils import run_gc
from ..worker import WorkerContext
from .base import BaseStage

logger = getLogger(__name__)

TAG_X4_V3 = "real-esrgan-x4-v3"


class UpscaleRealESRGANStage(BaseStage):
    def load(
        self, server: ServerContext, params: UpscaleParams, device: DeviceParams, tile=0
    ):
        # must be within load function for patches to take effect
        # TODO: rewrite and remove
        from realesrgan import RealESRGANer

        model_file = "%s.%s" % (params.upscale_model, params.format)
        model_path = path.join(server.model_path, model_file)

        cache_key = (model_path, params.format)
        cache_pipe = server.cache.get(ModelTypes.upscaling, cache_key)
        if cache_pipe is not None:
            logger.info("reusing existing Real ESRGAN pipeline")
            return cache_pipe

        if not path.isfile(model_path):
            raise FileNotFoundError("Real ESRGAN model not found at %s" % model_path)

        # TODO: swap for regular RRDBNet after rewriting wrapper
        model = OnnxRRDBNet(
            server,
            model_file,
            provider=device.ort_provider(),
            sess_options=device.sess_options(),
        )

        dni_weight = None
        if params.upscale_model == TAG_X4_V3 and params.denoise != 1:
            wdn_model_path = model_path.replace(TAG_X4_V3, "%s-wdn" % TAG_X4_V3)
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
            half=False,  # TODO: use server optimizations
        )

        server.cache.set(ModelTypes.upscaling, cache_key, upsampler)
        run_gc([device])

        return upsampler

    def run(
        self,
        worker: WorkerContext,
        server: ServerContext,
        stage: StageParams,
        _params: ImageParams,
        sources: List[Image.Image],
        *,
        upscale: UpscaleParams,
        stage_source: Optional[Image.Image] = None,
        **kwargs,
    ) -> List[Image.Image]:
        logger.info("upscaling image with Real ESRGAN: x%s", upscale.scale)

        outputs = []
        for source in sources:
            output = np.array(source)
            upsampler = self.load(
                server, upscale, worker.get_device(), tile=stage.tile_size
            )

            output, _ = upsampler.enhance(output, outscale=upscale.outscale)

            output = Image.fromarray(output, "RGB")
            logger.info("final output image size: %sx%s", output.width, output.height)
            outputs.append(output)

        return outputs
