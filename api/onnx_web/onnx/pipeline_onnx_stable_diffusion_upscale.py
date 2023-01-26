from diffusers import (
    DDPMScheduler,
    OnnxRuntimeModel,
    StableDiffusionUpscalePipeline,
)
from diffusers.models import (
    CLIPTokenizer,
)
from diffusers.schedulers import (
    KarrasDiffusionSchedulers,
)
from typing import (
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
        tokenizer: CLIPTokenizer,
        unet: OnnxRuntimeModel,
        low_res_scheduler: DDPMScheduler,
        scheduler: KarrasDiffusionSchedulers,
        max_noise_level: int = 350,
    ):
        super().__init__(vae, text_encoder, tokenizer, unet, low_res_scheduler, scheduler, max_noise_level)

    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        image: Union[torch.FloatTensor, PIL.Image.Image,
                     List[PIL.Image.Image]] = None,
        num_inference_steps: int = 75,
        guidance_scale: float = 9.0,
        noise_level: int = 20,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator,
                                  List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[
            int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
    ):
        pass
