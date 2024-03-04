import unittest

from onnx_web.prompt.base import PromptNetwork, PromptPhrase
from onnx_web.prompt.grammar import PhraseNode, TokenNode
from onnx_web.prompt.parser import compile_prompt_onnx, parse_prompt_onnx


class ParserTests(unittest.TestCase):
    def test_single_word_phrase(self):
        res = parse_prompt_onnx(None, "foo (bar) bin", debug=False)
        self.assertListEqual(
            res,
            [
                ["foo"],
                PhraseNode(["bar"], weight=1.5),
                ["bin"],
            ],
        )

    def test_multi_word_phrase(self):
        res = parse_prompt_onnx(None, "foo bar (middle words) bin bun", debug=False)
        self.assertListEqual(
            res,
            [
                ["foo", "bar"],
                PhraseNode(["middle", "words"], weight=1.5),
                ["bin", "bun"],
            ],
        )

    def test_nested_phrase(self):
        res = parse_prompt_onnx(None, "foo (((bar))) bin", debug=False)
        self.assertListEqual(
            res,
            [
                ["foo"],
                PhraseNode(["bar"], weight=(1.5**3)),
                ["bin"],
            ],
        )

    def test_clip_skip_token(self):
        res = parse_prompt_onnx(None, "foo <clip:skip:2> bin", debug=False)
        self.assertListEqual(
            res,
            [
                ["foo"],
                TokenNode("clip", "skip", 2),
                ["bin"],
            ],
        )

    def test_lora_token(self):
        res = parse_prompt_onnx(None, "foo <lora:name:1.5> bin", debug=False)
        self.assertListEqual(
            res,
            [
                ["foo"],
                TokenNode("lora", "name", 1.5),
                ["bin"],
            ],
        )

    def test_region_token(self):
        res = parse_prompt_onnx(
            None, "foo <region:1:2:3:4:0.5:0.75:prompt> bin", debug=False
        )
        self.assertListEqual(
            res,
            [
                ["foo"],
                TokenNode("region", None, [1, 2, 3, 4, 0.5, 0.75, ["prompt"]]),
                ["bin"],
            ],
        )

    def test_reseed_token(self):
        res = parse_prompt_onnx(None, "foo <reseed:1:2:3:4:12345> bin", debug=False)
        self.assertListEqual(
            res,
            [
                ["foo"],
                TokenNode("reseed", None, [1, 2, 3, 4, 12345]),
                ["bin"],
            ],
        )

    def test_compile_basic(self):
        prompt = compile_prompt_onnx("foo <clip:skip:2> bar (baz) <lora:qux:1.5>")
        self.assertEqual(prompt.networks, [PromptNetwork("lora", "qux", 1.5)])
        self.assertEqual(
            prompt.positive_phrases,
            [
                PromptPhrase("foo"),
                PromptPhrase("bar"),
                PromptPhrase(["baz"], weight=1.5),
            ],
        )
