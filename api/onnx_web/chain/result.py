from typing import List, Optional

import numpy as np
from PIL import Image


class StageResult:
    """
    Chain pipeline stage result.
    Can contain PIL images or numpy arrays, with helpers to convert between them.
    This class intentionally does not provide `__iter__`, to ensure clients get results in the format
    they are expected.
    """

    arrays: Optional[List[np.ndarray]]
    images: Optional[List[Image.Image]]

    @staticmethod
    def empty():
        return StageResult(images=[])

    @staticmethod
    def from_arrays(arrays: List[np.ndarray]):
        return StageResult(arrays=arrays)

    @staticmethod
    def from_images(images: List[Image.Image]):
        return StageResult(images=images)

    def __init__(self, arrays=None, images=None) -> None:
        if arrays is not None and images is not None:
            raise ValueError("stages must only return one type of result")
        elif arrays is None and images is None:
            raise ValueError("stages must return results")

        self.arrays = arrays
        self.images = images

    def __len__(self) -> int:
        if self.arrays is not None:
            return len(self.arrays)
        else:
            return len(self.images)

    def as_numpy(self) -> List[np.ndarray]:
        if self.arrays is not None:
            return self.arrays

        return [np.array(i) for i in self.images]

    def as_image(self) -> List[Image.Image]:
        if self.images is not None:
            return self.images

        return [Image.fromarray(np.uint8(i), shape_mode(i)) for i in self.arrays]


def shape_mode(arr: np.ndarray) -> str:
    if len(arr.shape) != 3:
        raise ValueError("unknown array format")

    if arr.shape[-1] == 3:
        return "RGB"
    elif arr.shape[-1] == 4:
        return "RGBA"

    raise ValueError("unknown image format")
