from diffusers import (
    DDPMScheduler,
    OnnxRuntimeModel,
    StableDiffusionUpscalePipeline,
)
from typing import (
    Any,
    Callable,
    Union,
    List,
    Optional,
)

import PIL
import torch


class OnnxStableDiffusionUpscalePipeline(StableDiffusionUpscalePipeline):
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
        super().__init__(vae, text_encoder, tokenizer, unet,
                         low_res_scheduler, scheduler, max_noise_level)

    def __call__(
        self,
        *args,
        **kwargs,
    ):
        super().__call__(*args, **kwargs)
