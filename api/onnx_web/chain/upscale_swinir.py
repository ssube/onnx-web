from logging import getLogger
from os import path
from typing import Optional

import numpy as np
from PIL import Image

from ..models.onnx import OnnxModel
from ..params import DeviceParams, ImageParams, StageParams, UpscaleParams
from ..server import ServerContext
from ..utils import run_gc
from ..worker import WorkerContext
from .stage import BaseStage

logger = getLogger(__name__)


class UpscaleSwinIRStage(BaseStage):
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
        cache_pipe = server.cache.get("swinir", cache_key)

        if cache_pipe is not None:
            logger.info("reusing existing SwinIR pipeline")
            return cache_pipe

        logger.debug("loading SwinIR model from %s", model_path)

        pipe = OnnxModel(
            server,
            model_path,
            provider=device.ort_provider(),
            sess_options=device.sess_options(),
        )

        server.cache.set("swinir", cache_key, pipe)
        run_gc([device])

        return pipe

    def run(
        self,
        job: WorkerContext,
        server: ServerContext,
        stage: StageParams,
        _params: ImageParams,
        source: Image.Image,
        *,
        upscale: UpscaleParams,
        stage_source: Optional[Image.Image] = None,
        **kwargs,
    ) -> Image.Image:
        upscale = upscale.with_args(**kwargs)
        source = stage_source or source

        if upscale.upscale_model is None:
            logger.warn("no correction model given, skipping")
            return source

        logger.info("correcting faces with SwinIR model: %s", upscale.upscale_model)
        device = job.get_device()
        swinir = self.load(server, stage, upscale, device)

        # TODO: add support for grayscale (1-channel) images
        image = np.array(source) / 255.0
        image = image[:, :, [2, 1, 0]].astype(np.float32).transpose((2, 0, 1))
        image = np.expand_dims(image, axis=0)
        logger.info("SwinIR input shape: %s", image.shape)

        scale = upscale.outscale
        dest = np.zeros(
            (
                image.shape[0],
                image.shape[1],
                image.shape[2] * scale,
                image.shape[3] * scale,
            )
        )
        logger.info("SwinIR output shape: %s", dest.shape)

        dest = swinir(image)
        dest = np.clip(np.squeeze(dest, axis=0), 0, 1)
        dest = dest[[2, 1, 0], :, :].transpose((1, 2, 0))
        dest = (dest * 255.0).round().astype(np.uint8)

        output = Image.fromarray(dest, "RGB")
        logger.info("output image size: %s x %s", output.width, output.height)
        return output
