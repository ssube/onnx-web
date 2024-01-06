from logging import getLogger
from os import path
from typing import Optional

from PIL import Image

from ..onnx import OnnxRRDBNet
from ..params import DeviceParams, ImageParams, StageParams, UpscaleParams
from ..server import ModelTypes, ServerContext
from ..utils import run_gc
from ..worker import WorkerContext
from .base import BaseStage
from .result import StageResult

logger = getLogger(__name__)

TAG_X4_V3 = "real-esrgan-x4-v3"


class UpscaleRealESRGANStage(BaseStage):
    def load(
        self, server: ServerContext, params: UpscaleParams, device: DeviceParams, tile=0
    ):
        # must be within load function for patches to take effect
        # TODO: rewrite and remove
        from realesrgan import RealESRGANer

        class RealESRGANWrapper(RealESRGANer):
            def __init__(
                self,
                scale,
                model_path,
                dni_weight=None,
                model=None,
                tile=0,
                tile_pad=10,
                pre_pad=10,
                half=False,
                device=None,
                gpu_id=None,
            ):
                self.scale = scale
                self.tile_size = tile
                self.tile_pad = tile_pad
                self.pre_pad = pre_pad
                self.mod_scale = None
                self.half = half
                self.model = model
                self.device = device

        model_file = f"{params.upscale_model}.onnx"
        model_path = path.join(server.model_path, model_file)

        cache_key = (model_path, params.scale)
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

        upsampler = RealESRGANWrapper(
            scale=params.scale,
            model_path=None,
            dni_weight=dni_weight,
            model=model,
            tile=tile,
            tile_pad=params.tile_pad,
            pre_pad=params.pre_pad,
            half=("torch-fp16" in server.optimizations),
            device=device.torch_str(),
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
        sources: StageResult,
        *,
        upscale: UpscaleParams,
        stage_source: Optional[Image.Image] = None,
        **kwargs,
    ) -> StageResult:
        logger.info("upscaling image with Real ESRGAN: x%s", upscale.scale)

        upsampler = self.load(
            server, upscale, worker.get_device(), tile=stage.tile_size
        )

        outputs = []
        for source in sources.as_arrays():
            output, _ = upsampler.enhance(source, outscale=upscale.outscale)
            logger.info("final output image size: %s", output.shape)
            outputs.append(output)

        return StageResult(arrays=outputs, metadata=sources.metadata)
