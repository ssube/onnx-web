from types import SimpleNamespace
from typing import List, Optional, Union

import numpy as np
import torch
from compel import Compel, ReturnedEmbeddingsType
from diffusers import OnnxStableDiffusionPipeline


def get_inference_session(model):
    if hasattr(model, "session"):
        return model.session

    if hasattr(model, "model"):
        return model.model

    raise ValueError("Model does not have an inference session")


def wrap_encoder(text_encoder, sdxl=False):
    class WrappedEncoder:
        device = "cpu"

        def __init__(self, text_encoder):
            self.text_encoder = text_encoder

        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)

        def forward(
            self, token_ids, attention_mask, output_hidden_states=None, return_dict=True
        ):
            """
            If `output_hidden_states` is None, return pooled embeds.
            """
            dtype = np.int32
            session = get_inference_session(self.text_encoder)
            if session.get_inputs()[0].type == "tensor(int64)":
                dtype = np.int64

            # TODO: does compel use attention masks?
            outputs = text_encoder(input_ids=token_ids.numpy().astype(dtype))

            if output_hidden_states is None:
                return SimpleNamespace(
                    text_embeds=torch.from_numpy(outputs[0]),
                    last_hidden_state=torch.from_numpy(outputs[1]),
                )
            elif output_hidden_states is True:
                hidden_states = [torch.from_numpy(state) for state in outputs[2:]]
                return SimpleNamespace(
                    last_hidden_state=torch.from_numpy(outputs[0]),
                    pooler_output=torch.from_numpy(outputs[1]),
                    hidden_states=hidden_states,
                )
            else:
                return SimpleNamespace(
                    last_hidden_state=torch.from_numpy(outputs[0]),
                    pooler_output=torch.from_numpy(outputs[1]),
                )

        def __getattr__(self, name):
            return getattr(self.text_encoder, name)

    return WrappedEncoder(text_encoder)


def encode_prompt_compel(
    self: OnnxStableDiffusionPipeline,
    prompt: str,
    num_images_per_prompt: int,
    do_classifier_free_guidance: bool,
    negative_prompt: Optional[str] = None,
    prompt_embeds: Optional[np.ndarray] = None,
    negative_prompt_embeds: Optional[np.ndarray] = None,
    skip_clip_states: int = 0,
) -> np.ndarray:
    embeddings_type = (
        ReturnedEmbeddingsType.LAST_HIDDEN_STATES_NORMALIZED
        if skip_clip_states == 0
        else ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NORMALIZED
    )
    wrapped_encoder = wrap_encoder(self.text_encoder)
    compel = Compel(
        tokenizer=self.tokenizer,
        text_encoder=wrapped_encoder,
        returned_embeddings_type=embeddings_type,
    )

    prompt_embeds = compel(prompt)

    if negative_prompt is not None:
        negative_prompt_embeds = compel(negative_prompt)

    if negative_prompt_embeds is not None:
        [prompt_embeds, negative_prompt_embeds] = (
            compel.pad_conditioning_tensors_to_same_length(
                [prompt_embeds, negative_prompt_embeds]
            )
        )

    prompt_embeds = prompt_embeds.numpy().astype(np.float32)
    if negative_prompt_embeds is not None:
        negative_prompt_embeds = negative_prompt_embeds.numpy().astype(np.float32)

    return np.concatenate([negative_prompt_embeds, prompt_embeds])


def encode_prompt_compel_sdxl(
    self: OnnxStableDiffusionPipeline,
    prompt: Union[str, List[str]],
    num_images_per_prompt: int,
    do_classifier_free_guidance: bool,
    negative_prompt: Optional[Union[str, list]],
    prompt_embeds: Optional[np.ndarray] = None,
    negative_prompt_embeds: Optional[np.ndarray] = None,
    pooled_prompt_embeds: Optional[np.ndarray] = None,
    negative_pooled_prompt_embeds: Optional[np.ndarray] = None,
    skip_clip_states: int = 0,
) -> np.ndarray:
    wrapped_encoder = wrap_encoder(self.text_encoder, sdxl=True)
    wrapped_encoder_2 = wrap_encoder(self.text_encoder_2, sdxl=True)
    compel = Compel(
        tokenizer=[self.tokenizer, self.tokenizer_2],
        text_encoder=[wrapped_encoder, wrapped_encoder_2],
        returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
        requires_pooled=[False, True],
    )

    prompt_embeds, prompt_pooled = compel(prompt)

    if negative_prompt is not None:
        negative_prompt_embeds, negative_pooled = compel(negative_prompt)

    if negative_prompt_embeds is not None:
        [prompt_embeds, negative_prompt_embeds] = (
            compel.pad_conditioning_tensors_to_same_length(
                [prompt_embeds, negative_prompt_embeds]
            )
        )

    prompt_embeds = prompt_embeds.numpy().astype(np.float32)
    prompt_pooled = prompt_pooled.numpy().astype(np.float32)
    if negative_prompt_embeds is not None:
        negative_prompt_embeds = negative_prompt_embeds.numpy().astype(np.float32)
        negative_pooled = negative_pooled.numpy().astype(np.float32)

    return (
        prompt_embeds,
        negative_prompt_embeds,
        prompt_pooled,
        negative_pooled,
    )
