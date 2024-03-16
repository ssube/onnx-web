from typing import Any, List, Union


def flatten(lst):
    for el in lst:
        if isinstance(el, list):
            yield from flatten(el)
        else:
            yield el


def collapse_phrases(
    nodes: List[Union[Any]],
    phrase,
    token,
) -> List[Union[Any]]:
    """
    Combine phrases with the same weight.
    """

    weight = None
    tokens = []
    phrases = []

    def flush_tokens():
        nonlocal weight, tokens
        if len(tokens) > 0:
            phrases.append(phrase(tokens, weight))
            tokens = []
            weight = None

    for node in nodes:
        if isinstance(node, str):
            node = phrase([node])
        elif isinstance(node, token):
            flush_tokens()
            phrases.append(node)
            continue

        if node.weight == weight:
            tokens.extend(node.tokens)
        else:
            flush_tokens()
            tokens = [*node.tokens]
            weight = node.weight

    flush_tokens()
    return phrases
