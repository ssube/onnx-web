from logging import getLogger
from typing import List, Optional

import numpy as np
from diffusers import OnnxRuntimeModel

from ...server import ServerContext

logger = getLogger(__name__)


class UNetWrapper(object):
    prompt_embeds: Optional[List[np.ndarray]] = None
    prompt_index: int = 0
    server: ServerContext
    wrapped: OnnxRuntimeModel
    xl: bool

    def __init__(
        self,
        server: ServerContext,
        wrapped: OnnxRuntimeModel,
        xl: bool,
    ):
        self.server = server
        self.wrapped = wrapped
        self.xl = xl

    def __call__(
        self,
        sample: np.ndarray = None,
        timestep: np.ndarray = None,
        encoder_hidden_states: np.ndarray = None,
        **kwargs,
    ):
        logger.trace(
            "UNet parameter types: %s, %s, %s",
            sample.dtype,
            timestep.dtype,
            encoder_hidden_states.dtype,
        )

        if self.prompt_embeds is not None:
            step_index = self.prompt_index % len(self.prompt_embeds)
            logger.trace("multiple prompt embeds found, using step: %s", step_index)
            encoder_hidden_states = self.prompt_embeds[step_index]
            self.prompt_index += 1

        if self.xl:
            logger.trace(
                "converting UNet sample to hidden state dtype for XL: %s",
                encoder_hidden_states.dtype,
            )
            sample = sample.astype(encoder_hidden_states.dtype)
        else:
            if sample.dtype != timestep.dtype:
                logger.trace("converting UNet sample to timestep dtype")
                sample = sample.astype(timestep.dtype)

            if encoder_hidden_states.dtype != timestep.dtype:
                logger.trace("converting UNet hidden states to timestep dtype")
                encoder_hidden_states = encoder_hidden_states.astype(timestep.dtype)

        return self.wrapped(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            **kwargs,
        )

    def __getattr__(self, attr):
        return getattr(self.wrapped, attr)

    def set_prompts(self, prompt_embeds: List[np.ndarray]):
        logger.debug(
            "setting prompt embeds for UNet: %s", [p.shape for p in prompt_embeds]
        )
        self.prompt_embeds = prompt_embeds
        self.prompt_index = 0
