from typing import List, Union

from arpeggio import EOF, OneOrMore, PTNodeVisitor, RegExMatch

from .utils import collapse_phrases, flatten


def token_delimiter():
    return ":"


def token():
    return RegExMatch(r"\w+")


def token_run():
    return OneOrMore(token)


def decimal():
    return RegExMatch(r"\d+\.\d*")


def integer():
    return RegExMatch(r"\d+")


def token_clip_skip():
    return ("clip", token_delimiter, "skip", token_delimiter, integer)


def token_inversion():
    return ("inversion", token_delimiter, token_run, token_delimiter, decimal)


def token_lora():
    return ("lora", token_delimiter, token_run, token_delimiter, decimal)


def token_region():
    return (
        "region",
        token_delimiter,
        integer,
        token_delimiter,
        integer,
        token_delimiter,
        integer,
        token_delimiter,
        integer,
        token_delimiter,
        decimal,
        token_delimiter,
        decimal,
        token_delimiter,
        token_run,
    )


def token_reseed():
    return (
        "reseed",
        token_delimiter,
        integer,
        token_delimiter,
        integer,
        token_delimiter,
        integer,
        token_delimiter,
        integer,
        token_delimiter,
        integer,
    )


def token_inner():
    return [token_clip_skip, token_inversion, token_lora, token_region, token_reseed]


def phrase_inner():
    return [phrase, token_run]


def pos_phrase():
    return ("(", OneOrMore(phrase_inner), ")")


def neg_phrase():
    return ("[", OneOrMore(phrase_inner), "]")


def token_phrase():
    return ("<", OneOrMore(token_inner), ">")


def phrase():
    return [pos_phrase, neg_phrase, token_phrase, token_run]


def prompt():
    return OneOrMore(phrase), EOF


class PhraseNode:
    def __init__(self, tokens: Union[List[str], str], weight: float = 1.0) -> None:
        self.tokens = tokens
        self.weight = weight

    def __repr__(self) -> str:
        return f"{self.tokens} * {self.weight}"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, self.__class__):
            return other.tokens == self.tokens and other.weight == self.weight

        return False


class TokenNode:
    def __init__(self, type: str, name: str, *rest):
        self.type = type
        self.name = name
        self.rest = rest

    def __repr__(self) -> str:
        return f"<{self.type}:{self.name}:{self.rest}>"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, self.__class__):
            return (
                other.type == self.type
                and other.name == self.name
                and other.rest == self.rest
            )

        return False


class OnnxPromptVisitor(PTNodeVisitor):
    def __init__(self, defaults=True, weight=0.5, **kwargs):
        super().__init__(defaults, **kwargs)

        self.neg_weight = weight
        self.pos_weight = 1.0 + weight

    def visit_decimal(self, node, children):
        return float(node.value)

    def visit_integer(self, node, children):
        return int(node.value)

    def visit_token(self, node, children):
        return str(node.value)

    def visit_token_clip_skip(self, node, children):
        return TokenNode("clip", "skip", children[0])

    def visit_token_inversion(self, node, children):
        return TokenNode("inversion", children[0][0], children[1])

    def visit_token_lora(self, node, children):
        return TokenNode("lora", children[0][0], children[1])

    def visit_token_region(self, node, children):
        return TokenNode("region", None, children)

    def visit_token_reseed(self, node, children):
        return TokenNode("reseed", None, children)

    def visit_token_run(self, node, children):
        return children

    def visit_phrase_inner(self, node, children):
        return [
            (
                child
                if isinstance(child, (PhraseNode, TokenNode, list))
                else PhraseNode(child)
            )
            for child in children
        ]

    def visit_pos_phrase(self, node, children):
        return parse_phrase(children, self.pos_weight)

    def visit_neg_phrase(self, node, children):
        return parse_phrase(children, self.neg_weight)

    def visit_phrase(self, node, children):
        return list(flatten(children))

    def visit_prompt(self, node, children):
        return collapse_phrases(list(flatten(children)), PhraseNode, TokenNode)


def parse_phrase(child, weight):
    if isinstance(child, PhraseNode):
        return PhraseNode(child.tokens, child.weight * weight)
    elif isinstance(child, str):
        return PhraseNode([child], weight)
    elif isinstance(child, list):
        # TODO: when this is a list of strings, create a single node with all of them
        # if all(isinstance(c, str) for c in child):
        #     return PhraseNode(child, weight)

        return [parse_phrase(c, weight) for c in child]
