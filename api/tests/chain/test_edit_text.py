import unittest

import numpy as np
from PIL import Image

from onnx_web.chain.edit_text import EditTextStage
from onnx_web.chain.result import StageResult


class TestEditTextStage(unittest.TestCase):
    def test_run(self):
        # Create a sample image
        image = Image.new("RGB", (100, 100), color="black")

        # Create an instance of EditTextStage
        stage = EditTextStage()

        # Define the input parameters
        text = "Hello, World!"
        position = (10, 10)
        fill = "white"
        stroke = "white"
        stroke_width = 2

        # Create a mock source StageResult
        source = StageResult.from_images([image], metadata={})

        # Call the run method
        result = stage.run(
            None,
            None,
            None,
            None,
            source,
            text=text,
            position=position,
            fill=fill,
            stroke=stroke,
            stroke_width=stroke_width,
        )

        # Assert the output
        self.assertEqual(len(result.as_images()), 1)
        # self.assertEqual(result.metadata, {})

        # Verify the modified image
        modified_image = result.as_images()[0]
        self.assertEqual(np.max(np.array(modified_image)), 255)
