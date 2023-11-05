import unittest

import numpy as np
import torch
from onnx import GraphProto, ModelProto
from onnx.numpy_helper import from_array, to_array

from onnx_web.convert.diffusion.textual_inversion import (
    blend_embedding_concept,
    blend_embedding_embeddings,
    blend_embedding_node,
    blend_embedding_parameters,
    blend_textual_inversions,
    detect_embedding_format,
)

TEST_DIMS = (8, 8)
TEST_DIMS_EMBEDS = (1, *TEST_DIMS)

TEST_MODEL_EMBEDS = {
      "string_to_token": {
        "test": 1,
      },
      "string_to_param": {
        "test": torch.from_numpy(np.ones(TEST_DIMS_EMBEDS)),
      },
}


class DetectEmbeddingFormatTests(unittest.TestCase):
  def test_concept(self):
    embedding = {
      "<test>": "test",
    }
    self.assertEqual(detect_embedding_format(embedding), "concept")

  def test_parameters(self):
    embedding = {
      "emb_params": "test",
    }
    self.assertEqual(detect_embedding_format(embedding), "parameters")

  def test_embeddings(self):
    embedding = {
      "string_to_token": "test",
      "string_to_param": "test",
    }
    self.assertEqual(detect_embedding_format(embedding), "embeddings")

  def test_unknown(self):
    embedding = {
      "what_is_this": "test",
    }
    self.assertEqual(detect_embedding_format(embedding), None)


class BlendEmbeddingConceptTests(unittest.TestCase):
  def test_existing_base_token(self):
    embeds = {
      "test": np.ones(TEST_DIMS),
    }
    blend_embedding_concept(embeds, {
      "<test>": torch.from_numpy(np.ones(TEST_DIMS)),
    }, np.float32, "test", 1.0)

    self.assertIn("test", embeds)
    self.assertEqual(embeds["test"].shape, TEST_DIMS)
    self.assertEqual(embeds["test"].mean(), 2)

  def test_missing_base_token(self):
    embeds = {}
    blend_embedding_concept(embeds, {
      "<test>": torch.from_numpy(np.ones(TEST_DIMS)),
    }, np.float32, "test", 1.0)

    self.assertIn("test", embeds)
    self.assertEqual(embeds["test"].shape, TEST_DIMS)

  def test_existing_token(self):
    embeds = {
      "<test>": np.ones(TEST_DIMS),
    }
    blend_embedding_concept(embeds, {
      "<test>": torch.from_numpy(np.ones(TEST_DIMS)),
    }, np.float32, "test", 1.0)

    keys = list(embeds.keys())
    keys.sort()

    self.assertIn("test", embeds)
    self.assertEqual(keys, ["<test>", "test"])

  def test_missing_token(self):
    embeds = {}
    blend_embedding_concept(embeds, {
      "<test>": torch.from_numpy(np.ones(TEST_DIMS)),
    }, np.float32, "test", 1.0)

    keys = list(embeds.keys())
    keys.sort()

    self.assertIn("test", embeds)
    self.assertEqual(keys, ["<test>", "test"])


class BlendEmbeddingParametersTests(unittest.TestCase):
  def test_existing_base_token(self):
    embeds = {
      "test": np.ones(TEST_DIMS),
    }
    blend_embedding_parameters(embeds, {
      "emb_params": torch.from_numpy(np.ones(TEST_DIMS_EMBEDS)),
    }, np.float32, "test", 1.0)

    self.assertIn("test", embeds)
    self.assertEqual(embeds["test"].shape, TEST_DIMS)
    self.assertEqual(embeds["test"].mean(), 2)

  def test_missing_base_token(self):
    embeds = {}
    blend_embedding_parameters(embeds, {
      "emb_params": torch.from_numpy(np.ones(TEST_DIMS_EMBEDS)),
    }, np.float32, "test", 1.0)

    self.assertIn("test", embeds)
    self.assertEqual(embeds["test"].shape, TEST_DIMS)

  def test_existing_token(self):
    embeds = {
      "test": np.ones(TEST_DIMS_EMBEDS),
    }
    blend_embedding_parameters(embeds, {
      "emb_params": torch.from_numpy(np.ones(TEST_DIMS_EMBEDS)),
    }, np.float32, "test", 1.0)

    keys = list(embeds.keys())
    keys.sort()

    self.assertIn("test", embeds)
    self.assertEqual(keys, ["test", "test-0", "test-all"])

  def test_missing_token(self):
    embeds = {}
    blend_embedding_parameters(embeds, {
      "emb_params": torch.from_numpy(np.ones(TEST_DIMS_EMBEDS)),
    }, np.float32, "test", 1.0)

    keys = list(embeds.keys())
    keys.sort()

    self.assertIn("test", embeds)
    self.assertEqual(keys, ["test", "test-0", "test-all"])


class BlendEmbeddingEmbeddingsTests(unittest.TestCase):
  def test_existing_base_token(self):
    embeds = {
      "test": np.ones(TEST_DIMS),
    }
    blend_embedding_embeddings(embeds, TEST_MODEL_EMBEDS, np.float32, "test", 1.0)

    self.assertIn("test", embeds)
    self.assertEqual(embeds["test"].shape, TEST_DIMS)
    self.assertEqual(embeds["test"].mean(), 2)

  def test_missing_base_token(self):
    embeds = {}
    blend_embedding_embeddings(embeds, TEST_MODEL_EMBEDS, np.float32, "test", 1.0)

    self.assertIn("test", embeds)
    self.assertEqual(embeds["test"].shape, TEST_DIMS)

  def test_existing_token(self):
    embeds = {
      "test": np.ones(TEST_DIMS),
    }
    blend_embedding_embeddings(embeds, TEST_MODEL_EMBEDS, np.float32, "test", 1.0)

    keys = list(embeds.keys())
    keys.sort()

    self.assertIn("test", embeds)
    self.assertEqual(keys, ["test", "test-0", "test-all"])

  def test_missing_token(self):
    embeds = {}
    blend_embedding_embeddings(embeds, TEST_MODEL_EMBEDS, np.float32, "test", 1.0)

    keys = list(embeds.keys())
    keys.sort()

    self.assertIn("test", embeds)
    self.assertEqual(keys, ["test", "test-0", "test-all"])


class BlendEmbeddingNodeTests(unittest.TestCase):
  def test_expand_weights(self):
    weights = from_array(np.ones(TEST_DIMS))
    weights.name = "text_model.embeddings.token_embedding.weight"

    model = ModelProto(graph=GraphProto(initializer=[
      weights,
    ]))

    embeds = {}
    blend_embedding_node(model, {
      'convert_tokens_to_ids': lambda t: t,
    }, embeds, 2)

    result = to_array(model.graph.initializer[0])

    self.assertEqual(len(model.graph.initializer), 1)
    self.assertEqual(result.shape, (10, 8)) # (8 + 2, 8)


class BlendTextualInversionsTests(unittest.TestCase):
  def test_blend_multi_concept(self):
    pass

  def test_blend_multi_parameters(self):
    pass

  def test_blend_multi_embeddings(self):
    pass

  def test_blend_multi_mixed(self):
    pass
