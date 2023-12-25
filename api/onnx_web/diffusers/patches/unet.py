from logging import getLogger
from typing import Dict, List, Optional, Union

import numpy as np
from diffusers import OnnxRuntimeModel
from optimum.onnxruntime.modeling_diffusion import ORTModelUnet

from ...server import ServerContext

logger = getLogger(__name__)


class UNetWrapper(object):
    input_types: Optional[Dict[str, np.dtype]] = None
    prompt_embeds: Optional[List[np.ndarray]] = None
    prompt_index: int = 0
    server: ServerContext
    wrapped: Union[OnnxRuntimeModel, ORTModelUnet]
    xl: bool

    def __init__(
        self,
        server: ServerContext,
        wrapped: Union[OnnxRuntimeModel, ORTModelUnet],
        xl: bool,
    ):
        self.server = server
        self.wrapped = wrapped
        self.xl = xl

        self.cache_input_types()

    def __call__(
        self,
        sample: Optional[np.ndarray] = None,
        timestep: Optional[np.ndarray] = None,
        encoder_hidden_states: Optional[np.ndarray] = None,
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

        if self.input_types is None:
            self.cache_input_types()

        if encoder_hidden_states.dtype != self.input_types["encoder_hidden_states"]:
            logger.trace("converting UNet hidden states to input dtype")
            encoder_hidden_states = encoder_hidden_states.astype(
                self.input_types["encoder_hidden_states"]
            )

        if sample.dtype != self.input_types["sample"]:
            logger.trace("converting UNet sample to input dtype")
            sample = sample.astype(self.input_types["sample"])

        if timestep.dtype != self.input_types["timestep"]:
            logger.trace("converting UNet timestep to input dtype")
            timestep = timestep.astype(self.input_types["timestep"])

        return self.wrapped(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            **kwargs,
        )

    def __getattr__(self, attr):
        return getattr(self.wrapped, attr)

    def cache_input_types(self):
        # TODO: use server dtype as default
        if isinstance(self.wrapped, ORTModelUnet):
            session = self.wrapped.session
        elif isinstance(self.wrapped, OnnxRuntimeModel):
            session = self.wrapped.model
        else:
            raise ValueError()

        inputs = session.get_inputs()
        self.input_types = dict([(input.name, input.type) for input in inputs])
        logger.debug("cached UNet input types: %s", self.input_types)

        # [
        #        (
        #            input.name,
        #            next(
        #                [
        #                    TENSOR_TYPE_TO_NP_TYPE[field[1].elem_type]
        #                    for field in input.type.ListFields()
        #                ],
        #                np.float32,
        #            ),
        #        )
        #        for input in self.wrapped.model.graph.input
        #    ]

    def set_prompts(self, prompt_embeds: List[np.ndarray]):
        logger.debug(
            "setting prompt embeds for UNet: %s", [p.shape for p in prompt_embeds]
        )
        self.prompt_embeds = prompt_embeds
        self.prompt_index = 0
