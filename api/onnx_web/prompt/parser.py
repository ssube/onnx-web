from typing import Literal, Union

import numpy as np
from arpeggio import ParserPython, visit_parse_tree

from .base import Prompt, PromptNetwork, PromptPhrase, PromptRegion, PromptSeed
from .grammar import OnnxPromptVisitor, PhraseNode, TokenNode
from .grammar import prompt as prompt_base


def parse_prompt_compel(pipeline, prompt: str) -> np.ndarray:
    from compel import Compel

    parser = Compel(tokenizer=pipeline.tokenizer, text_encoder=pipeline.text_encoder)
    return parser([prompt])


def parse_prompt_lpw(pipeline, prompt: str, debug=False) -> np.ndarray:
    raise NotImplementedError()


def parse_prompt_onnx(pipeline, prompt: str, debug=False) -> np.ndarray:
    parser = ParserPython(prompt_base, debug=debug)
    visitor = OnnxPromptVisitor()

    lst = parser.parse(prompt)
    ast = visit_parse_tree(lst, visitor)

    return ast


def parse_prompt_vanilla(pipeline, prompt: str) -> np.ndarray:
    return pipeline._encode_prompt(prompt)


def parse_prompt(
    pipeline,
    prompt: str,
    engine: Literal["compel", "lpw", "onnx-web", "pipeline"] = "onnx-web",
) -> np.ndarray:
    if engine == "compel":
        return parse_prompt_compel(pipeline, prompt)
    if engine == "lpw":
        return parse_prompt_lpw(pipeline, prompt)
    elif engine == "onnx-web":
        return parse_prompt_onnx(pipeline, prompt)
    elif engine == "pipeline":
        return parse_prompt_vanilla(pipeline, prompt)
    else:
        raise ValueError("invalid prompt parser")


def compile_prompt_onnx(prompt: str) -> Prompt:
    ast = parse_prompt_onnx(None, prompt)

    tokens = [node for node in ast if isinstance(node, TokenNode)]
    networks = [
        PromptNetwork(token.type, token.name, token.rest[0])
        for token in tokens
        if token.type in ["lora", "inversion"]
    ]
    regions = [PromptRegion(*token.rest) for token in tokens if token.type == "region"]
    reseeds = [PromptSeed(*token.rest) for token in tokens if token.type == "reseed"]

    phrases = [
        compile_prompt_phrase(node)
        for node in ast
        if isinstance(node, (list, PhraseNode, str))
    ]
    phrases = list(flatten(phrases))

    return Prompt(
        networks=networks,
        positive_phrases=phrases,
        negative_phrases=[],
        region_prompts=regions,
        region_seeds=reseeds,
    )


def compile_prompt_phrase(node: Union[PhraseNode, str]) -> PromptPhrase:
    if isinstance(node, list):
        return [compile_prompt_phrase(subnode) for subnode in node]

    if isinstance(node, str):
        return PromptPhrase(node)

    return PromptPhrase(node.tokens, node.weight)


def flatten(val):
    if isinstance(val, list):
        for subval in val:
            yield from flatten(subval)
    else:
        yield val
