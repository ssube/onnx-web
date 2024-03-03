from typing import List, Union

from arpeggio import EOF, OneOrMore, PTNodeVisitor, RegExMatch


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
    return [token_inversion, token_lora, token_region, token_reseed]


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


class PromptPhrase:
    def __init__(self, tokens: Union[List[str], str], weight: float = 1.0) -> None:
        self.tokens = tokens
        self.weight = weight

    def __repr__(self) -> str:
        return f"{self.tokens} * {self.weight}"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, self.__class__):
            return other.tokens == self.tokens and other.weight == self.weight

        return False


class PromptToken:
    def __init__(self, token_type: str, token_name: str, *rest):
        self.token_type = token_type
        self.token_name = token_name
        self.rest = rest

    def __repr__(self) -> str:
        return f"<{self.token_type}:{self.token_name}:{self.rest}>"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, self.__class__):
            return (
                other.token_type == self.token_type
                and other.token_name == self.token_name
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

    def visit_token_inversion(self, node, children):
        return PromptToken("inversion", children[0][0], children[1])

    def visit_token_lora(self, node, children):
        return PromptToken("lora", children[0][0], children[1])

    def visit_token_region(self, node, children):
        return PromptToken("region", None, children)

    def visit_token_reseed(self, node, children):
        return PromptToken("reseed", None, children)

    def visit_token_run(self, node, children):
        return children

    def visit_phrase_inner(self, node, children):
        if isinstance(children[0], PromptPhrase):
            return children[0]
        elif isinstance(children[0], PromptToken):
            return children[0]
        else:
            return PromptPhrase(children[0])

    def visit_pos_phrase(self, node, children):
        c = children[0]
        if isinstance(c, PromptPhrase):
            return PromptPhrase(c.tokens, c.weight * self.pos_weight)
        elif isinstance(c, str):
            return PromptPhrase(c, self.pos_weight)

    def visit_neg_phrase(self, node, children):
        c = children[0]
        if isinstance(c, PromptPhrase):
            return PromptPhrase(c.tokens, c.weight * self.neg_weight)
        elif isinstance(c, str):
            return PromptPhrase(c, self.neg_weight)

    def visit_phrase(self, node, children):
        return children[0]

    def visit_prompt(self, node, children):
        return children
