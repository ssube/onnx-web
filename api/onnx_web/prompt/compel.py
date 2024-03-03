from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch
from compel import Compel, ReturnedEmbeddingsType
from diffusers import OnnxStableDiffusionPipeline


def wrap_encoder(text_encoder):
    class WrappedEncoder:
        device = "cpu"

        def __init__(self, text_encoder):
            self.text_encoder = text_encoder

        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)

        def forward(
            self, token_ids, attention_mask, output_hidden_states=True, return_dict=True
        ):
            # TODO: does compel use attention masks?
            outputs = text_encoder(input_ids=token_ids.numpy().astype(np.int32))
            if return_dict:
                if output_hidden_states:
                    hidden_states = outputs[2:]
                    return SimpleNamespace(
                        last_hidden_state=torch.from_numpy(outputs[0]),
                        pooler_output=torch.from_numpy(outputs[1]),
                        hidden_states=torch.from_numpy(hidden_states),
                    )
                else:
                    return SimpleNamespace(
                        last_hidden_state=torch.from_numpy(outputs[0]),
                        pooler_output=torch.from_numpy(outputs[1]),
                    )
            else:
                return outputs

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
    wrapped_encoder = wrap_encoder(self.text_encoder)
    compel = Compel(tokenizer=self.tokenizer, text_encoder=wrapped_encoder)

    prompt_embeds = compel(prompt)

    if negative_prompt is not None:
        negative_prompt_embeds = compel(self, negative_prompt)

    if negative_prompt_embeds is not None:
        [prompt_embeds, negative_prompt_embeds] = (
            compel.pad_conditioning_tensors_to_same_length(
                [prompt_embeds, negative_prompt_embeds]
            )
        )

    prompt_embeds = prompt_embeds.numpy().astype(np.int32)
    if negative_prompt_embeds is not None:
        negative_prompt_embeds = negative_prompt_embeds.numpy().astype(np.int32)

    return np.concatenate([negative_prompt_embeds, prompt_embeds])


def encode_prompt_compel_sdxl(
    self: OnnxStableDiffusionPipeline,
    prompt: str,
    num_images_per_prompt: int,
    do_classifier_free_guidance: bool,
    negative_prompt: Optional[str] = None,
    prompt_embeds: Optional[np.ndarray] = None,
    negative_prompt_embeds: Optional[np.ndarray] = None,
    skip_clip_states: int = 0,
) -> np.ndarray:
    wrapped_encoder = wrap_encoder(self.text_encoder)
    wrapped_encoder_2 = wrap_encoder(self.text_encoder_2)
    compel = Compel(
        tokenizer=[self.tokenizer, self.tokenizer_2],
        text_encoder=[wrapped_encoder, wrapped_encoder_2],
        returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
        requires_pooled=[False, True],
    )

    prompt_embeds, prompt_pooled = compel(prompt)

    if negative_prompt is not None:
        negative_prompt_embeds, negative_pooled = compel(self, negative_prompt)

    if negative_prompt_embeds is not None:
        [prompt_embeds, negative_prompt_embeds] = (
            compel.pad_conditioning_tensors_to_same_length(
                [prompt_embeds, negative_prompt_embeds]
            )
        )

    prompt_embeds = prompt_embeds.numpy().astype(np.int32)
    prompt_pooled = prompt_pooled.numpy().astype(np.int32)
    if negative_prompt_embeds is not None:
        negative_prompt_embeds = negative_prompt_embeds.numpy().astype(np.int32)
        negative_pooled = negative_pooled.numpy().astype(np.int32)

    return (
        prompt_embeds,
        negative_prompt_embeds,
        prompt_pooled,
        negative_pooled,
    )
