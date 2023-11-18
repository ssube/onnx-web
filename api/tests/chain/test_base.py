import unittest

from onnx_web.chain.pipeline import ChainProgress


class ChainProgressTests(unittest.TestCase):
    def test_accumulate_with_reset(self):
        def parent(step, timestep, latents):
            pass

        progress = ChainProgress(parent)
        progress(5, 1, None)
        progress(0, 1, None)
        progress(5, 1, None)

        self.assertEqual(progress.get_total(), 10)

    def test_start_value(self):
        def parent(step, timestep, latents):
            pass

        progress = ChainProgress(parent, 5)
        self.assertEqual(progress.get_total(), 5)

        progress(10, 1, None)
        self.assertEqual(progress.get_total(), 10)
