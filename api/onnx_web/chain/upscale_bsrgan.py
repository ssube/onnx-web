from logging import getLogger
from os import path
from typing import List, Optional

import numpy as np
from PIL import Image

from ..models.onnx import OnnxModel
from ..params import DeviceParams, ImageParams, Size, StageParams, UpscaleParams
from ..server import ModelTypes, ServerContext
from ..utils import run_gc
from ..worker import WorkerContext
from .base import BaseStage

logger = getLogger(__name__)


class UpscaleBSRGANStage(BaseStage):
    max_tile = 64

    def load(
        self,
        server: ServerContext,
        _stage: StageParams,
        upscale: UpscaleParams,
        device: DeviceParams,
    ):
        # must be within the load function for patch to take effect
        model_path = path.join(server.model_path, "%s.onnx" % (upscale.upscale_model))
        cache_key = (model_path,)
        cache_pipe = server.cache.get(ModelTypes.upscaling, cache_key)

        if cache_pipe is not None:
            logger.debug("reusing existing BSRGAN pipeline")
            return cache_pipe

        logger.info("loading BSRGAN model from %s", model_path)

        pipe = OnnxModel(
            server,
            model_path,
            provider=device.ort_provider(),
            sess_options=device.sess_options(),
        )

        server.cache.set(ModelTypes.upscaling, cache_key, pipe)
        run_gc([device])

        return pipe

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
        upscale = upscale.with_args(**kwargs)

        if upscale.upscale_model is None:
            logger.warning("no upscaling model given, skipping")
            return sources

        logger.info("upscaling with BSRGAN model: %s", upscale.upscale_model)
        device = worker.get_device()
        bsrgan = self.load(server, stage, upscale, device)

        outputs = []
        for source in sources:
            image = np.array(source) / 255.0
            image = image[:, :, [2, 1, 0]].astype(np.float32).transpose((2, 0, 1))
            image = np.expand_dims(image, axis=0)
            logger.trace("BSRGAN input shape: %s", image.shape)

            scale = upscale.outscale
            dest = np.zeros(
                (
                    image.shape[0],
                    image.shape[1],
                    image.shape[2] * scale,
                    image.shape[3] * scale,
                )
            )
            logger.trace("BSRGAN output shape: %s", dest.shape)

            dest = bsrgan(image)

            dest = np.clip(np.squeeze(dest, axis=0), 0, 1)
            dest = dest[[2, 1, 0], :, :].transpose((1, 2, 0))
            dest = (dest * 255.0).round().astype(np.uint8)

            output = Image.fromarray(dest, "RGB")
            logger.debug("output image size: %s x %s", output.width, output.height)

            outputs.append(output)

        return outputs

    def steps(
        self,
        params: ImageParams,
        size: Size,
    ) -> int:
        tile = min(params.unet_tile, self.max_tile)
        return size.width // tile * size.height // tile
