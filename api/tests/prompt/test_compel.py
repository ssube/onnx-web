import unittest
from unittest.mock import MagicMock

import numpy as np

from onnx_web.prompt.compel import (
    encode_prompt_compel,
    encode_prompt_compel_sdxl,
    get_inference_session,
    wrap_encoder,
)


class TestCompelHelpers(unittest.TestCase):
    def test_get_inference_session_missing(self):
        self.assertRaises(ValueError, get_inference_session, None)

    def test_get_inference_session_onnx_session(self):
        model = MagicMock()
        model.model = None
        model.session = "session"
        self.assertEqual(get_inference_session(model), "session")

    def test_get_inference_session_onnx_model(self):
        model = MagicMock()
        model.model = "model"
        model.session = None
        self.assertEqual(get_inference_session(model), "model")

    def test_wrap_encoder(self):
        text_encoder = MagicMock()
        wrapped = wrap_encoder(text_encoder)
        self.assertEqual(wrapped.device, "cpu")
        self.assertEqual(wrapped.text_encoder, text_encoder)


class TestCompelEncodePrompt(unittest.TestCase):
    def test_encode_basic(self):
        pipeline = MagicMock()
        pipeline.text_encoder = MagicMock()
        pipeline.text_encoder.return_value = [
            np.array([[1], [2]]),
            np.array([[3], [4]]),
        ]
        pipeline.tokenizer = MagicMock()
        pipeline.tokenizer.model_max_length = 1

        embeds = encode_prompt_compel(pipeline, "prompt", 1, True)
        np.testing.assert_equal(embeds, [[[3, 3]], [[3, 3]]])


class TestCompelEncodePromptSDXL(unittest.TestCase):
    @unittest.skip("need to fix the tensor shapes")
    def test_encode_basic(self):
        text_encoder_output = MagicMock()
        text_encoder_output.hidden_states = [[0], [1], [2], [3]]

        def call_text_encoder(*args, return_dict=False, **kwargs):
            print("call_text_encoder", return_dict)
            if return_dict:
                return text_encoder_output

            return [np.array([[1]]), np.array([[3]]), np.array([[5]]), np.array([[7]])]

        pipeline = MagicMock()
        pipeline.text_encoder.side_effect = call_text_encoder
        pipeline.text_encoder_2.side_effect = call_text_encoder
        pipeline.tokenizer.model_max_length = 1
        pipeline.tokenizer_2.model_max_length = 1

        embeds = encode_prompt_compel_sdxl(pipeline, "prompt", 1, True)
        np.testing.assert_equal(embeds, [[[3, 3]], [[3, 3]]])


if __name__ == "__main__":
    unittest.main()
