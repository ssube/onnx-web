from typing import Literal

import numpy as np
from arpeggio import ParserPython, visit_parse_tree

from .grammar import OnnxPromptVisitor
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

    ast = parser.parse(prompt)
    return visit_parse_tree(ast, visitor)


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
