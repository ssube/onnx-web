from logging import getLogger
from math import ceil
from re import Pattern, compile
from typing import List, Optional, Tuple

import numpy as np
from diffusers import OnnxStableDiffusionPipeline

logger = getLogger(__name__)


INVERSION_TOKEN = compile(r"\<inversion:(\w+):([\.|\d]+)\>")
LORA_TOKEN = compile(r"\<lora:(\w+):([\.|\d]+)\>")
MAX_TOKENS_PER_GROUP = 77
PATTERN_RANGE = compile(r"(\w+)-{(\d+),(\d+)(?:,(\d+))?}")


def expand_prompt_ranges(prompt: str) -> str:
    def expand_range(match):
        (base_token, start, end, step) = match.groups(default=1)
        num_tokens = [
            f"{base_token}-{i}" for i in range(int(start), int(end), int(step))
        ]
        return " ".join(num_tokens)

    return PATTERN_RANGE.sub(expand_range, prompt)


def expand_prompt(
    self: OnnxStableDiffusionPipeline,
    prompt: str,
    num_images_per_prompt: int,
    do_classifier_free_guidance: bool,
    negative_prompt: Optional[str] = None,
) -> "np.NDArray":
    # self provides:
    #   tokenizer: CLIPTokenizer
    #   encoder: OnnxRuntimeModel

    batch_size = len(prompt) if isinstance(prompt, list) else 1
    prompt = expand_prompt_ranges(prompt)

    # split prompt into 75 token chunks
    tokens = self.tokenizer(
        prompt,
        padding="max_length",
        return_tensors="np",
        max_length=self.tokenizer.model_max_length,
        truncation=False,
    )

    groups_count = ceil(tokens.input_ids.shape[1] / MAX_TOKENS_PER_GROUP)
    logger.debug("splitting %s into %s groups", tokens.input_ids.shape, groups_count)

    groups = []
    # np.array_split(tokens.input_ids, groups_count, axis=1)
    for i in range(groups_count):
        group_start = i * MAX_TOKENS_PER_GROUP
        group_end = min(
            group_start + MAX_TOKENS_PER_GROUP, tokens.input_ids.shape[1]
        )  # or should this be 1?
        logger.debug("building group for token slice [%s : %s]", group_start, group_end)
        groups.append(tokens.input_ids[:, group_start:group_end])

    # encode each chunk
    logger.debug("group token shapes: %s", [t.shape for t in groups])
    group_embeds = []
    for group in groups:
        logger.debug("encoding group: %s", group.shape)
        embeds = self.text_encoder(input_ids=group.astype(np.int32))[0]
        group_embeds.append(embeds)

    # concat those embeds
    logger.debug("group embeds shape: %s", [t.shape for t in group_embeds])
    prompt_embeds = np.concatenate(group_embeds, axis=1)
    prompt_embeds = np.repeat(prompt_embeds, num_images_per_prompt, axis=0)

    # get unconditional embeddings for classifier free guidance
    if do_classifier_free_guidance:
        uncond_tokens: List[str]
        if negative_prompt is None:
            uncond_tokens = [""] * batch_size
        elif type(prompt) is not type(negative_prompt):
            raise TypeError(
                f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                f" {type(prompt)}."
            )
        elif isinstance(negative_prompt, str):
            uncond_tokens = [negative_prompt] * batch_size
        elif batch_size != len(negative_prompt):
            raise ValueError(
                f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                " the batch size of `prompt`."
            )
        else:
            uncond_tokens = negative_prompt

        uncond_input = self.tokenizer(
            uncond_tokens,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="np",
        )
        negative_prompt_embeds = self.text_encoder(
            input_ids=uncond_input.input_ids.astype(np.int32)
        )[0]
        negative_padding = tokens.input_ids.shape[1] - negative_prompt_embeds.shape[1]
        logger.debug(
            "padding negative prompt to match input: %s, %s, %s extra tokens",
            tokens.input_ids.shape,
            negative_prompt_embeds.shape,
            negative_padding,
        )
        negative_prompt_embeds = np.pad(
            negative_prompt_embeds,
            [(0, 0), (0, negative_padding), (0, 0)],
            mode="constant",
            constant_values=0,
        )
        negative_prompt_embeds = np.repeat(
            negative_prompt_embeds, num_images_per_prompt, axis=0
        )

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        prompt_embeds = np.concatenate([negative_prompt_embeds, prompt_embeds])

    logger.debug("expanded prompt shape: %s", prompt_embeds.shape)
    return prompt_embeds


def get_tokens_from_prompt(
    prompt: str, pattern: Pattern
) -> Tuple[str, List[Tuple[str, float]]]:
    """
    TODO: replace with Arpeggio
    """
    remaining_prompt = prompt

    tokens = []
    next_match = pattern.search(remaining_prompt)
    while next_match is not None:
        logger.debug("found token in prompt: %s", next_match)
        name, weight = next_match.groups()
        tokens.append((name, float(weight)))
        # remove this match and look for another
        remaining_prompt = (
            remaining_prompt[: next_match.start()]
            + remaining_prompt[next_match.end() :]
        )
        next_match = pattern.search(remaining_prompt)

    return (remaining_prompt, tokens)


def get_loras_from_prompt(prompt: str) -> Tuple[str, List[Tuple[str, float]]]:
    return get_tokens_from_prompt(prompt, LORA_TOKEN)


def get_inversions_from_prompt(prompt: str) -> Tuple[str, List[Tuple[str, float]]]:
    return get_tokens_from_prompt(prompt, INVERSION_TOKEN)
