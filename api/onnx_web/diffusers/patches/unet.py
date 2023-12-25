from logging import getLogger
from typing import Dict, List, Optional, Union

import numpy as np
from diffusers import OnnxRuntimeModel
from diffusers.pipelines.onnx_utils import ORT_TO_NP_TYPE
from optimum.onnxruntime.modeling_diffusion import ORTModelUnet

from ...server import ServerContext

logger = getLogger(__name__)


class UNetWrapper(object):
    input_types: Optional[Dict[str, np.dtype]] = None
    prompt_embeds: Optional[List[np.ndarray]] = None
    prompt_index: int = 0
    sample_dtype: np.dtype
    server: ServerContext
    timestep_dtype: np.dtype
    wrapped: Union[OnnxRuntimeModel, ORTModelUnet]
    xl: bool

    def __init__(
        self,
        server: ServerContext,
        wrapped: Union[OnnxRuntimeModel, ORTModelUnet],
        xl: bool,
        sample_dtype: Optional[np.dtype] = None,
        timestep_dtype: np.dtype = np.int64,
    ):
        self.server = server
        self.wrapped = wrapped
        self.xl = xl
        self.sample_dtype = sample_dtype or server.torch_dtype
        self.timestep_dtype = timestep_dtype

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

        encoder_hidden_states_input_dtype = self.input_types.get(
            "encoder_hidden_states", self.sample_dtype
        )
        if encoder_hidden_states.dtype != encoder_hidden_states_input_dtype:
            logger.debug(
                "converting UNet hidden states to input dtype from %s to %s",
                encoder_hidden_states.dtype,
                encoder_hidden_states_input_dtype,
            )
            encoder_hidden_states = encoder_hidden_states.astype(
                encoder_hidden_states_input_dtype
            )

        sample_input_dtype = self.input_types.get("sample", self.sample_dtype)
        if sample.dtype != sample_input_dtype:
            logger.debug(
                "converting UNet sample to input dtype from %s to %s",
                sample.dtype,
                sample_input_dtype,
            )
            sample = sample.astype(sample_input_dtype)

        timestep_input_dtype = self.input_types.get("timestep", self.timestep_dtype)
        if timestep.dtype != timestep_input_dtype:
            logger.debug(
                "converting UNet timestep to input dtype from %s to %s",
                timestep.dtype,
                timestep_input_dtype,
            )
            timestep = timestep.astype(timestep_input_dtype)

        return self.wrapped(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            **kwargs,
        )

    def __getattr__(self, attr):
        return getattr(self.wrapped, attr)

    def cache_input_types(self):
        if isinstance(self.wrapped, ORTModelUnet):
            session = self.wrapped.session
        elif isinstance(self.wrapped, OnnxRuntimeModel):
            session = self.wrapped.model
        else:
            raise ValueError("unknown UNet class")

        inputs = session.get_inputs()
        self.input_types = dict(
            [(input.name, ORT_TO_NP_TYPE[input.type]) for input in inputs]
        )
        logger.debug("cached UNet input types: %s", self.input_types)

    def set_prompts(self, prompt_embeds: List[np.ndarray]):
        logger.debug(
            "setting prompt embeds for UNet: %s", [p.shape for p in prompt_embeds]
        )
        self.prompt_embeds = prompt_embeds
        self.prompt_index = 0
