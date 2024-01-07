import unittest
from unittest.mock import MagicMock

from onnx_web.chain.edit_metadata import EditMetadataStage


class TestEditMetadataStage(unittest.TestCase):
    def setUp(self):
        self.stage = EditMetadataStage()

    def test_run_with_no_changes(self):
        source = MagicMock()
        source.metadata = []

        result = self.stage.run(None, None, None, None, source)

        self.assertEqual(result, source)

    def test_run_with_note_change(self):
        source = MagicMock()
        source.metadata = [MagicMock()]
        note = "New note"

        result = self.stage.run(None, None, None, None, source, note=note)

        self.assertEqual(result, source)
        self.assertEqual(result.metadata[0].note, note)

    def test_run_with_replace_params_change(self):
        source = MagicMock()
        source.metadata = [MagicMock()]
        replace_params = MagicMock()

        result = self.stage.run(
            None, None, None, None, source, replace_params=replace_params
        )

        self.assertEqual(result, source)
        self.assertEqual(result.metadata[0].params, replace_params)

    # Add more test cases for other parameters...
