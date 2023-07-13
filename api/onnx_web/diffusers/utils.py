import random
from copy import deepcopy
from logging import getLogger
from math import ceil
from re import Pattern, compile
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from diffusers import OnnxStableDiffusionPipeline

from ..params import ImageParams, Size

logger = getLogger(__name__)

LATENT_CHANNELS = 4
LATENT_FACTOR = 8
MAX_TOKENS_PER_GROUP = 77

CLIP_TOKEN = compile(r"\<clip:([-\w]+):(\d+)\>")
INVERSION_TOKEN = compile(r"\<inversion:([^:\>]+):(-?[\.|\d]+)\>")
LORA_TOKEN = compile(r"\<lora:([^:\>]+):(-?[\.|\d]+)\>")
WILDCARD_TOKEN = compile(r"__([-/\\\w]+)__")

INTERVAL_RANGE = compile(r"(\w+)-{(\d+),(\d+)(?:,(\d+))?}")
ALTERNATIVE_RANGE = compile(r"\(([^\)]+)\)")


def expand_interval_ranges(prompt: str) -> str:
    def expand_range(match):
        (base_token, start, end, step) = match.groups(default=1)
        num_tokens = [
            f"{base_token}-{i}" for i in range(int(start), int(end), int(step))
        ]
        return " ".join(num_tokens)

    return INTERVAL_RANGE.sub(expand_range, prompt)


def expand_alternative_ranges(prompt: str) -> List[str]:
    prompt_groups = []

    last_end = 0
    next_group = ALTERNATIVE_RANGE.search(prompt)
    while next_group is not None:
        logger.debug("found alternative group in prompt: %s", next_group)

        if next_group.start() > last_end:
            skipped_prompt = prompt[last_end : next_group.start()]
            logger.trace("appending skipped section of prompt: %s", skipped_prompt)
            prompt_groups.append([skipped_prompt])

        options = next_group.group()[1:-1].split("|")
        logger.trace("split up alternative options: %s", options)
        prompt_groups.append(options)

        last_end = next_group.end()
        next_group = ALTERNATIVE_RANGE.search(prompt, last_end)

    if last_end < len(prompt):
        remaining_prompt = prompt[last_end:]
        logger.trace("appending remainder of prompt: %s", remaining_prompt)
        prompt_groups.append([remaining_prompt])

    prompt_count = max([len(group) for group in prompt_groups])
    prompts = []
    for i in range(prompt_count):
        options = []
        for group in prompt_groups:
            group_i = i % len(group)
            options.append(group[group_i])

        prompts.append("".join(options))

    return prompts


@torch.no_grad()
def expand_prompt(
    self: OnnxStableDiffusionPipeline,
    prompt: str,
    num_images_per_prompt: int,
    do_classifier_free_guidance: bool,
    negative_prompt: Optional[str] = None,
    prompt_embeds: Optional[np.ndarray] = None,
    negative_prompt_embeds: Optional[np.ndarray] = None,
    skip_clip_states: Optional[int] = 0,
) -> "np.NDArray":
    # self provides:
    #   tokenizer: CLIPTokenizer
    #   encoder: OnnxRuntimeModel

    prompt, clip_tokens = get_tokens_from_prompt(prompt, CLIP_TOKEN)
    if len(clip_tokens) > 0:
        skip_clip_states = int(clip_tokens[0][1])
        logger.info("skipping %s CLIP layers", skip_clip_states)

    batch_size = len(prompt) if isinstance(prompt, list) else 1
    prompt = expand_interval_ranges(prompt)

    # split prompt into 75 token chunks
    tokens = self.tokenizer(
        prompt,
        padding="max_length",
        return_tensors="np",
        max_length=self.tokenizer.model_max_length,
        truncation=False,
    )

    groups_count = ceil(tokens.input_ids.shape[1] / MAX_TOKENS_PER_GROUP)
    logger.trace("splitting %s into %s groups", tokens.input_ids.shape, groups_count)

    groups = []
    # np.array_split(tokens.input_ids, groups_count, axis=1)
    for i in range(groups_count):
        group_start = i * MAX_TOKENS_PER_GROUP
        group_end = min(
            group_start + MAX_TOKENS_PER_GROUP, tokens.input_ids.shape[1]
        )  # or should this be 1?
        logger.trace("building group for token slice [%s : %s]", group_start, group_end)

        group_size = group_end - group_start
        if group_size < MAX_TOKENS_PER_GROUP:
            pass  # TODO: pad short groups

        groups.append(tokens.input_ids[:, group_start:group_end])

    # encode each chunk
    logger.trace("group token shapes: %s", [t.shape for t in groups])
    group_embeds = []
    for group in groups:
        logger.trace("encoding group: %s", group.shape)

        text_result = self.text_encoder(input_ids=group.astype(np.int32))
        logger.trace(
            "text encoder produced %s outputs: %s",
            len(text_result),
            [t.shape for t in text_result],
        )

        last_state, _pooled_output, *hidden_states = text_result
        if skip_clip_states > 0:
            layer_norm = torch.nn.LayerNorm(last_state.shape[2])
            norm_state = layer_norm(
                torch.from_numpy(
                    hidden_states[-skip_clip_states].astype(np.float32)
                ).detach()
            )
            logger.trace(
                "normalized results after skipping %s layers: %s",
                skip_clip_states,
                norm_state.shape,
            )
            group_embeds.append(
                norm_state.numpy().astype(hidden_states[-skip_clip_states].dtype)
            )
        else:
            group_embeds.append(last_state)

    # concat those embeds
    logger.trace("group embeds shape: %s", [t.shape for t in group_embeds])
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
        logger.trace(
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

    logger.trace("expanded prompt shape: %s", prompt_embeds.shape)
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


def get_latents_from_seed(seed: int, size: Size, batch: int = 1) -> np.ndarray:
    """
    From https://www.travelneil.com/stable-diffusion-updates.html.
    """
    latents_shape = (
        batch,
        LATENT_CHANNELS,
        size.height // LATENT_FACTOR,
        size.width // LATENT_FACTOR,
    )
    rng = np.random.default_rng(seed)
    image_latents = rng.standard_normal(latents_shape).astype(np.float32)
    return image_latents


def get_tile_latents(
    full_latents: np.ndarray,
    seed: int,
    size: Size,
    dims: Tuple[int, int, int],
) -> np.ndarray:
    x, y, tile = dims
    t = tile // LATENT_FACTOR
    x = max(0, x // LATENT_FACTOR)
    y = max(0, y // LATENT_FACTOR)
    xt = x + t
    yt = y + t

    mx = size.width // LATENT_FACTOR
    my = size.height // LATENT_FACTOR

    tile_latents = full_latents[:, :, y:yt, x:xt]

    if tile_latents.shape[2] < t or tile_latents.shape[3] < t:
        extra_latents = get_latents_from_seed(seed, size, batch=tile_latents.shape[0])
        extra_latents[:, :, 0:tile_latents.shape[2], 0:tile_latents.shape[3]] = tile_latents
        tile_latents = extra_latents

    return tile_latents


def get_scaled_latents(
    seed: int,
    size: Size,
    batch: int = 1,
    scale: int = 1,
) -> np.ndarray:
    latents = get_latents_from_seed(seed, size, batch=batch)
    latents = torch.from_numpy(latents)

    scaled = torch.nn.functional.interpolate(
        latents, scale_factor=(scale, scale), mode="bilinear"
    )
    return scaled.numpy()


def parse_prompt(
    params: ImageParams,
    use_input: bool = False,
) -> Tuple[
    List[Tuple[str, str]],
    List[Tuple[str, float]],
    List[Tuple[str, float]],
    Tuple[str, str],
]:
    prompt, loras = get_loras_from_prompt(
        params.input_prompt if use_input else params.prompt
    )
    prompt, inversions = get_inversions_from_prompt(prompt)
    # params.prompt = prompt

    neg_prompt = None
    if params.input_negative_prompt is not None:
        neg_prompt, neg_loras = get_loras_from_prompt(
            params.input_negative_prompt if use_input else params.negative_prompt
        )
        neg_prompt, neg_inversions = get_inversions_from_prompt(neg_prompt)
        # params.negative_prompt = neg_prompt

        loras.extend(neg_loras)
        inversions.extend(neg_inversions)

    prompts = expand_alternative_ranges(prompt)
    if neg_prompt is not None:
        neg_prompts = expand_alternative_ranges(neg_prompt)
    else:
        neg_prompts = [None] * len(prompts)

    logger.trace("generated prompts: %s, %s", prompts, neg_prompts)

    # count these ahead of time, because they will change
    prompt_count = len(prompts)
    neg_prompt_count = len(neg_prompts)

    if prompt_count < neg_prompt_count:
        # extend prompts
        for i in range(prompt_count, neg_prompt_count):
            prompts.append(prompts[i % prompt_count])
    elif prompt_count > neg_prompt_count:
        # extend neg_prompts
        for i in range(neg_prompt_count, prompt_count):
            neg_prompts.append(neg_prompts[i % neg_prompt_count])

    return list(zip(prompts, neg_prompts)), loras, inversions, (prompt, neg_prompt)


def encode_prompt(
    pipe: OnnxStableDiffusionPipeline,
    prompt_pairs: List[Tuple[str, str]],
    num_images_per_prompt: int = 1,
    do_classifier_free_guidance: bool = True,
) -> List[np.ndarray]:
    return [
        pipe._encode_prompt(
            prompt,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=neg_prompt,
        )
        for prompt, neg_prompt in prompt_pairs
    ]


def parse_wildcards(prompt: str, seed: int, wildcards: Dict[str, List[str]]) -> str:
    next_match = WILDCARD_TOKEN.search(prompt)
    remaining_prompt = prompt

    # prep a local copy to avoid mutating the main one
    wildcards = deepcopy(wildcards)
    random.seed(seed)

    while next_match is not None:
        logger.debug("found wildcard in prompt: %s", next_match)
        name, *rest = next_match.groups()

        wildcard = ""
        if name in wildcards:
            wildcard = pop_random(wildcards.get(name))
        else:
            logger.warning("unknown wildcard: %s", name)

        remaining_prompt = (
            remaining_prompt[: next_match.start()]
            + wildcard
            + remaining_prompt[next_match.end() :]
        )
        next_match = WILDCARD_TOKEN.search(remaining_prompt)

    return remaining_prompt


def replace_wildcards(params: ImageParams, wildcards: Dict[str, List[str]]):
    params.prompt = parse_wildcards(params.prompt, params.seed, wildcards)
    if params.negative_prompt is not None:
        params.negative_prompt = parse_wildcards(
            params.negative_prompt, params.seed, wildcards
        )


def pop_random(list: List[str]) -> str:
    """
    From https://stackoverflow.com/a/14088129
    """
    i = random.randrange(len(list))
    list[i], list[-1] = list[-1], list[i]
    return list.pop()
