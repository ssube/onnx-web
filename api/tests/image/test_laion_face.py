import unittest

import numpy as np

from onnx_web.image.laion_face import draw_pupils, generate_annotation, reverse_channels


class TestLaionFace(unittest.TestCase):
    @unittest.skip("need to prepare a good input image")
    def test_draw_pupils(self):
        # Create a dummy image
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        # Create a dummy landmark list
        class LandmarkList:
            def __init__(self, landmarks):
                self.landmark = landmarks

        # Create a dummy drawing spec
        class DrawingSpec:
            def __init__(self, color):
                self.color = color

        # Create some dummy landmarks
        landmarks = [
            # Add your landmarks here
        ]

        # Create a dummy drawing spec
        drawing_spec = DrawingSpec(color=(255, 0, 0))  # Red color

        # Call the draw_pupils function
        draw_pupils(image, LandmarkList(landmarks), drawing_spec)

        self.assertNotEqual(np.sum(image), 0, "Image should be modified")

    @unittest.skip("need to prepare a good input image")
    def test_generate_annotation(self):
        # Create a dummy image
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        # Call the generate_annotation function
        result = generate_annotation(image, max_faces=1, min_confidence=0.5)

        self.assertEqual(
            result.shape,
            image.shape,
            "Result shape should be the same as the input image",
        )
        self.assertNotEqual(np.sum(result), 0, "Result should not be all zeros")


class TestReverseChannels(unittest.TestCase):
    def test_reverse_channels(self):
        # Create a dummy image
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        layer = np.ones((100, 100), dtype=np.uint8)
        image[:, :, 0] = layer

        # Call the reverse_channels function
        reversed_image = reverse_channels(image)

        self.assertEqual(
            image.shape, reversed_image.shape, "Image shape should be the same"
        )
        self.assertTrue(
            np.array_equal(reversed_image[:, :, 2], layer),
            "Channels should be reversed",
        )
