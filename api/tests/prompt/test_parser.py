import unittest

from onnx_web.prompt.grammar import PromptPhrase, PromptToken
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
            ],
        )

    def test_multi_word_phrase(self):
        res = parse_prompt_onnx(None, "foo bar (middle words) bin bun", debug=False)
        self.assertListEqual(
            [str(i) for i in res],
            [
                str(["foo", "bar"]),
                str(PromptPhrase(["middle", "words"], weight=1.5)),
                str(["bin", "bun"]),
            ],
        )

    def test_nested_phrase(self):
        res = parse_prompt_onnx(None, "foo (((bar))) bin", debug=False)
        self.assertListEqual(
            [str(i) for i in res],
            [
                str(["foo"]),
                str(PromptPhrase(["bar"], weight=(1.5**3))),
                str(["bin"]),
            ],
        )

    def test_lora_token(self):
        res = parse_prompt_onnx(None, "foo <lora:name:1.5> bin", debug=False)
        self.assertListEqual(
            [str(i) for i in res],
            [
                str(["foo"]),
                str(PromptToken("lora", "name", 1.5)),
                str(["bin"]),
            ],
        )

    def test_region_token(self):
        res = parse_prompt_onnx(
            None, "foo <region:1:2:3:4:0.5:0.75:prompt> bin", debug=False
        )
        self.assertListEqual(
            [str(i) for i in res],
            [
                str(["foo"]),
                str(PromptToken("region", None, [1, 2, 3, 4, 0.5, 0.75, ["prompt"]])),
                str(["bin"]),
            ],
        )

    def test_reseed_token(self):
        res = parse_prompt_onnx(None, "foo <reseed:1:2:3:4:12345> bin", debug=False)
        self.assertListEqual(
            [str(i) for i in res],
            [
                str(["foo"]),
                str(PromptToken("reseed", None, [1, 2, 3, 4, 12345])),
                str(["bin"]),
            ],
        )
