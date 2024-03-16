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
                PhraseNode(["foo"]),
                PhraseNode(["bar"], weight=1.5),
                PhraseNode(["bin"]),
            ],
        )

    def test_multi_word_phrase(self):
        res = parse_prompt_onnx(None, "foo bar (middle words) bin bun", debug=False)
        self.assertListEqual(
            res,
            [
                PhraseNode(["foo", "bar"]),
                PhraseNode(["middle", "words"], weight=1.5),
                PhraseNode(["bin", "bun"]),
            ],
        )

    def test_nested_phrase(self):
        res = parse_prompt_onnx(None, "foo (((bar))) bin", debug=False)
        self.assertListEqual(
            res,
            [
                PhraseNode(["foo"]),
                PhraseNode(["bar"], weight=(1.5**3)),
                PhraseNode(["bin"]),
            ],
        )

    def test_clip_skip_token(self):
        res = parse_prompt_onnx(None, "foo <clip:skip:2> bin", debug=False)
        self.assertListEqual(
            res,
            [
                PhraseNode(["foo"]),
                TokenNode("clip", "skip", 2),
                PhraseNode(["bin"]),
            ],
        )

    def test_lora_token(self):
        res = parse_prompt_onnx(None, "foo <lora:name:1.5> bin", debug=False)
        self.assertListEqual(
            res,
            [
                PhraseNode(["foo"]),
                TokenNode("lora", "name", 1.5),
                PhraseNode(["bin"]),
            ],
        )

    def test_region_token(self):
        res = parse_prompt_onnx(
            None, "foo <region:1:2:3:4:0.5:0.75:prompt> bin", debug=False
        )
        self.assertListEqual(
            res,
            [
                PhraseNode(["foo"]),
                TokenNode("region", None, [1, 2, 3, 4, 0.5, 0.75, ["prompt"]]),
                PhraseNode(["bin"]),
            ],
        )

    def test_reseed_token(self):
        res = parse_prompt_onnx(None, "foo <reseed:1:2:3:4:12345> bin", debug=False)
        self.assertListEqual(
            res,
            [
                PhraseNode(["foo"]),
                TokenNode("reseed", None, [1, 2, 3, 4, 12345]),
                PhraseNode(["bin"]),
            ],
        )

    def test_compile_tokens(self):
        prompt = compile_prompt_onnx("foo <clip:skip:2> bar (baz) <lora:qux:1.5>")

        self.assertEqual(prompt.clip_skip, 2)
        self.assertEqual(prompt.networks, [PromptNetwork("lora", "qux", 1.5)])
        self.assertEqual(
            prompt.positive_phrases,
            [
                PromptPhrase(["foo"]),
                PromptPhrase(["bar"]),
                PromptPhrase(["baz"], weight=1.5),
            ],
        )

    def test_compile_weights(self):
        prompt = compile_prompt_onnx("foo ((bar)) baz [[qux]] bun ([nest] me)")

        self.assertEqual(
            prompt.positive_phrases,
            [
                PromptPhrase(["foo"]),
                PromptPhrase(["bar"], weight=2.25),
                PromptPhrase(["baz"]),
                PromptPhrase(["qux"], weight=0.25),
                PromptPhrase(["bun"]),
                PromptPhrase(["nest"], weight=0.75),
                PromptPhrase(["me"], weight=1.5),
            ],
        )

    def test_compile_runs(self):
        prompt = compile_prompt_onnx("foo <clip:skip:2> bar (baz) <lora:qux:1.5>")
        prompt.collapse_runs()

        self.assertEqual(
            prompt.positive_phrases,
            [
                PromptPhrase(["foo bar"]),
                PromptPhrase(["baz"], weight=1.5),
            ],
        )
