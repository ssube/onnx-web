from typing import Any, List, Optional

import numpy as np
from PIL import Image

from ..output import json_params
from ..params import Border, HighresParams, ImageParams, Size, UpscaleParams


class ImageMetadata:
    border: Border
    highres: HighresParams
    params: ImageParams
    size: Size
    upscale: UpscaleParams

    def __init__(
        self,
        params: ImageParams,
        size: Size,
        upscale: Optional[UpscaleParams] = None,
        border: Optional[Border] = None,
        highres: Optional[HighresParams] = None,
    ) -> None:
        self.params = params
        self.size = size
        self.upscale = upscale
        self.border = border
        self.highres = highres

    def tojson(self):
        return json_params(
            [],
            self.params,
            self.size,
            upscale=self.upscale,
            border=self.border,
            highres=self.highres,
        )


class StageResult:
    """
    Chain pipeline stage result.
    Can contain PIL images or numpy arrays, with helpers to convert between them.
    This class intentionally does not provide `__iter__`, to ensure clients get results in the format
    they are expected.
    """

    arrays: Optional[List[np.ndarray]]
    images: Optional[List[Image.Image]]
    metadata: List[ImageMetadata]

    @staticmethod
    def empty():
        return StageResult(images=[])

    @staticmethod
    def from_arrays(arrays: List[np.ndarray]):
        return StageResult(arrays=arrays)

    @staticmethod
    def from_images(images: List[Image.Image]):
        return StageResult(images=images)

    def __init__(
        self,
        arrays: Optional[List[np.ndarray]] = None,
        images: Optional[List[Image.Image]] = None,
        source: Optional[Any] = None,
    ) -> None:
        if sum([arrays is not None, images is not None, source is not None]) > 1:
            raise ValueError("stages must only return one type of result")
        elif arrays is None and images is None and source is None:
            raise ValueError("stages must return results")

        self.arrays = arrays
        self.images = images
        self.source = source

    def __len__(self) -> int:
        if self.arrays is not None:
            return len(self.arrays)
        elif self.images is not None:
            return len(self.images)
        else:
            return 0

    def as_numpy(self) -> List[np.ndarray]:
        if self.arrays is not None:
            return self.arrays
        elif self.images is not None:
            return [np.array(i) for i in self.images]
        else:
            return []

    def as_image(self) -> List[Image.Image]:
        if self.images is not None:
            return self.images
        elif self.arrays is not None:
            return [Image.fromarray(np.uint8(i), shape_mode(i)) for i in self.arrays]
        else:
            return []

    def push_array(self, array: np.ndarray, metadata: Optional[ImageMetadata]):
        if self.arrays is not None:
            self.arrays.append(array)
        elif self.images is not None:
            self.images.append(Image.fromarray(np.uint8(array), shape_mode(array)))
        else:
            raise ValueError("invalid stage result")

        if metadata is not None:
            self.metadata.append(metadata)
        else:
            self.metadata.append(ImageMetadata())

    def push_image(self, image: Image.Image, metadata: Optional[ImageMetadata]):
        if self.images is not None:
            self.images.append(image)
        elif self.arrays is not None:
            self.arrays.append(np.array(image))
        else:
            raise ValueError("invalid stage result")

        if metadata is not None:
            self.metadata.append(metadata)
        else:
            self.metadata.append(ImageMetadata())


def shape_mode(arr: np.ndarray) -> str:
    if len(arr.shape) != 3:
        raise ValueError("unknown array format")

    if arr.shape[-1] == 3:
        return "RGB"
    elif arr.shape[-1] == 4:
        return "RGBA"

    raise ValueError("unknown image format")
