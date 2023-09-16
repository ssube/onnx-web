import unittest

import numpy as np

from onnx_web.diffusers.utils import (
  expand_interval_ranges,
  expand_alternative_ranges,
  get_inversions_from_prompt,
  get_latents_from_seed,
  get_loras_from_prompt,
  get_scaled_latents,
  get_tokens_from_prompt,
)
from onnx_web.params import Size

class TestExpandIntervalRanges(unittest.TestCase):
  def test_prompt_with_no_ranges(self):
    prompt = "an astronaut eating a hamburger"
    result = expand_interval_ranges(prompt)
    self.assertEqual(prompt, result)

  def test_prompt_with_range(self):
    prompt = "an astronaut-{1,4} eating a hamburger"
    result = expand_interval_ranges(prompt)
    self.assertEqual(result, "an astronaut-1 astronaut-2 astronaut-3 eating a hamburger")

class TestExpandAlternativeRanges(unittest.TestCase):
  def test_prompt_with_no_ranges(self):
    prompt = "an astronaut eating a hamburger"
    result = expand_alternative_ranges(prompt)
    self.assertEqual([prompt], result)

  def test_ranges_match(self):
    prompt = "(an astronaut|a squirrel) eating (a hamburger|an acorn)"
    result = expand_alternative_ranges(prompt)
    self.assertEqual(result, ["an astronaut eating a hamburger", "a squirrel eating an acorn"])

class TestInversionsFromPrompt(unittest.TestCase):
  def test_get_inversions(self):
    prompt = "<inversion:test:1.0> an astronaut eating an embedding"
    result, tokens = get_inversions_from_prompt(prompt)

    self.assertEqual(result, " an astronaut eating an embedding")
    self.assertEqual(tokens, [("test", 1.0)])

class TestLoRAsFromPrompt(unittest.TestCase):
  def test_get_loras(self):
    prompt = "<lora:test:1.0> an astronaut eating a LoRA"
    result, tokens = get_loras_from_prompt(prompt)

    self.assertEqual(result, " an astronaut eating a LoRA")
    self.assertEqual(tokens, [("test", 1.0)])

class TestLatentsFromSeed(unittest.TestCase):
  def test_batch_size(self):
    latents = get_latents_from_seed(1, Size(64, 64), batch=4)
    self.assertEqual(latents.shape, (4, 4, 8, 8))

  def test_consistency(self):
    latents1 = get_latents_from_seed(1, Size(64, 64))
    latents2 = get_latents_from_seed(1, Size(64, 64))
    self.assertTrue(np.array_equal(latents1, latents2))

class TestTileLatents(unittest.TestCase):
  def test_full_tile(self):
    pass

  def test_partial_tile(self):
    pass

class TestScaledLatents(unittest.TestCase):
  def test_scale_up(self):
    latents = get_latents_from_seed(1, Size(16, 16))
    scaled = get_scaled_latents(1, Size(16, 16), scale=2)
    self.assertEqual(latents[0, 0, 0, 0], scaled[0, 0, 0, 0])

  def test_scale_down(self):
    latents = get_latents_from_seed(1, Size(16, 16))
    scaled = get_scaled_latents(1, Size(16, 16), scale=0.5)
    self.assertEqual((
      latents[0, 0, 0, 0] +
      latents[0, 0, 0, 1] +
      latents[0, 0, 1, 0] +
      latents[0, 0, 1, 1]
    ) / 4, scaled[0, 0, 0, 0])
