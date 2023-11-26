from logging import getLogger
from typing import Any, List

from diffusers.pipelines.onnx_utils import OnnxRuntimeModel
from diffusers.pipelines.stable_diffusion import OnnxStableDiffusionUpscalePipeline as BasePipeline
from diffusers.schedulers import DDPMScheduler

logger = getLogger(__name__)


class FakeConfig:
    block_out_channels: List[int]
    scaling_factor: float

    def __init__(self) -> None:
        self.block_out_channels = [128, 256, 512]
        self.scaling_factor = 0.08333


class OnnxStableDiffusionUpscalePipeline(BasePipeline):
    def __init__(
        self,
        vae: OnnxRuntimeModel,
        text_encoder: OnnxRuntimeModel,
        tokenizer: Any,
        unet: OnnxRuntimeModel,
        low_res_scheduler: DDPMScheduler,
        scheduler: Any,
        max_noise_level: int = 350,
    ):
        if not hasattr(vae, "config"):
            setattr(vae, "config", FakeConfig())

        super().__init__(
            vae,
            text_encoder,
            tokenizer,
            unet,
            low_res_scheduler,
            scheduler,
            max_noise_level=max_noise_level,
        )
