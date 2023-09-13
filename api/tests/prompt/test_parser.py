import unittest

from onnx_web.prompt.grammar import PromptPhrase
from onnx_web.prompt.parser import parse_prompt_onnx


class ParserTests(unittest.TestCase):
    def test_single_word_phrase(self):
        res = parse_prompt_onnx(None, "foo (bar) bin", debug=False)
        self.assertListEqual(
            [str(i) for i in res],
            [
                str(["foo"]),
                str(PromptPhrase(["bar"], weight=1.5)),
                str(["bin"]),
            ]
        )

    def test_multi_word_phrase(self):
        res = parse_prompt_onnx(None, "foo bar (middle words) bin bun", debug=False)
        self.assertListEqual(
            [str(i) for i in res],
            [
                str(["foo", "bar"]),
                str(PromptPhrase(["middle", "words"], weight=1.5)),
                str(["bin", "bun"]),
            ]
        )

    def test_nested_phrase(self):
        res = parse_prompt_onnx(None, "foo (((bar))) bin", debug=False)
        self.assertListEqual(
            [str(i) for i in res],
            [
                str(["foo"]),
                str(PromptPhrase(["bar"], weight=(1.5 ** 3))),
                str(["bin"]),
            ]
        )
