from typing import List, Union

from arpeggio import EOF, OneOrMore, PTNodeVisitor, RegExMatch


def token():
    return RegExMatch(r"\w+")


def token_run():
    return OneOrMore(token)


def phrase_inner():
    return [phrase, token_run]


def pos_phrase():
    return ("(", OneOrMore(phrase_inner), ")")


def neg_phrase():
    return ("[", OneOrMore(phrase_inner), "]")


def phrase():
    return [pos_phrase, neg_phrase, token_run]


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


class OnnxPromptVisitor(PTNodeVisitor):
    def __init__(self, defaults=True, weight=0.5, **kwargs):
        super().__init__(defaults, **kwargs)

        self.neg_weight = weight
        self.pos_weight = 1.0 + weight

    def visit_token(self, node, children):
        return str(node.value)

    def visit_token_run(self, node, children):
        return children

    def visit_phrase_inner(self, node, children):
        if isinstance(children[0], PromptPhrase):
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
