from logging import getLogger
from os import path
from typing import List, Optional

import numpy as np
from PIL import Image

from ..params import DeviceParams, ImageParams, StageParams, UpscaleParams
from ..server import ModelTypes, ServerContext
from ..utils import run_gc
from ..worker import WorkerContext
from .base import BaseStage

logger = getLogger(__name__)


class CorrectGFPGANStage(BaseStage):
    def load(
        self,
        server: ServerContext,
        _stage: StageParams,
        upscale: UpscaleParams,
        device: DeviceParams,
    ):
        # must be within the load function for patch to take effect
        # TODO: rewrite and remove
        from gfpgan import GFPGANer

        face_path = path.join(server.cache_path, "%s.pth" % (upscale.correction_model))
        cache_key = (face_path,)
        cache_pipe = server.cache.get(ModelTypes.correction, cache_key)

        if cache_pipe is not None:
            logger.info("reusing existing GFPGAN pipeline")
            return cache_pipe

        logger.debug("loading GFPGAN model from %s", face_path)

        # TODO: find a way to pass the ONNX model to underlying architectures
        gfpgan = GFPGANer(
            arch="clean",
            bg_upsampler=None,
            channel_multiplier=2,
            device=device.torch_str(),
            model_path=face_path,
            upscale=upscale.face_outscale,
        )

        server.cache.set(ModelTypes.correction, cache_key, gfpgan)
        run_gc([device])

        return gfpgan

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

        if upscale.correction_model is None:
            logger.warning("no face model given, skipping")
            return sources

        logger.info("correcting faces with GFPGAN model: %s", upscale.correction_model)
        device = worker.get_device()
        gfpgan = self.load(server, stage, upscale, device)

        outputs = []
        for source in sources:
            output = np.array(source)
            _, _, output = gfpgan.enhance(
                output,
                has_aligned=False,
                only_center_face=False,
                paste_back=True,
                weight=upscale.face_strength,
            )
            outputs.append(Image.fromarray(output, "RGB"))

        return outputs
