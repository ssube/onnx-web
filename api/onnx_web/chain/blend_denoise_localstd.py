from logging import getLogger
from typing import Optional

import numpy as np
from PIL import Image

from ..params import ImageParams, SizeChart, StageParams
from ..server import ServerContext
from ..worker import ProgressCallback, WorkerContext
from .base import BaseStage
from .result import StageResult

logger = getLogger(__name__)


class BlendDenoiseLocalStdStage(BaseStage):
    max_tile = SizeChart.max

    def run(
        self,
        _worker: WorkerContext,
        _server: ServerContext,
        _stage: StageParams,
        _params: ImageParams,
        sources: StageResult,
        *,
        strength: int = 3,
        stage_source: Optional[Image.Image] = None,
        callback: Optional[ProgressCallback] = None,
        **kwargs,
    ) -> StageResult:
        logger.info("denoising source images")

        results = []
        for source in sources.as_numpy():
            results.append(remove_noise(source))

        return StageResult(arrays=results)


def downscale_image(image):
    result_image = np.zeros((image.shape[0] // 2, image.shape[1] // 2), dtype=np.uint8)

    for i in range(0, image.shape[0] - 1, 2):
        for j in range(0, image.shape[1] - 1, 2):
            # Average the four neighboring pixels
            pixel_average = np.mean(image[i : i + 2, j : j + 2], axis=(0, 1))
            result_image[i // 2, j // 2] = pixel_average.astype(np.uint8)

    return result_image


def replace_noise(region, threshold):
    # Identify stray pixels (brightness significantly deviates from surrounding pixels)
    central_pixel = region[1, 1]

    region_median = np.median(region)
    region_deviation = np.std(region)
    diff = np.abs(central_pixel - region_median)

    # If the whole region is fairly consistent but the central pixel deviates significantly,
    if diff > region_deviation and diff > threshold:
        surrounding_pixels = region[region != central_pixel]
        surrounding_median = np.median(surrounding_pixels)
        # replace it with the median of surrounding pixels
        region[1, 1] = surrounding_median
        return True

    return False


def remove_noise(image, region_size=(6, 6), threshold=10):
    # Assuming 'image' is a 3D numpy array representing the RGB image

    # Create a copy of the original image to store the result
    result_image = np.copy(image)
    # result_mask = np.ones_like(image) * 255

    # Iterate over regions in each channel
    i_inc = region_size[0] // 2
    j_inc = region_size[1] // 2

    for i in range(i_inc, image.shape[0] - i_inc, 1):
        for j in range(j_inc, image.shape[1] - j_inc, 1):
            i_min = i - (region_size[0] // 2)
            i_max = i + (region_size[0] // 2)
            j_min = j - (region_size[1] // 2)
            j_max = j + (region_size[1] // 2)

            # Extract region from each channel
            region_red = downscale_image(image[i_min:i_max, j_min:j_max, 0])
            region_green = downscale_image(image[i_min:i_max, j_min:j_max, 1])
            region_blue = downscale_image(image[i_min:i_max, j_min:j_max, 2])

            replaced = any(
                [
                    replace_noise(region_red, threshold),
                    replace_noise(region_green, threshold),
                ]
            )

            # Apply the noise removal function to each channel
            if replaced:
                # Assign the processed region back to the result image
                result_image[i - 1 : i + 1, j - 1 : j + 1, 0] = region_red[1, 1]
                result_image[i - 1 : i + 1, j - 1 : j + 1, 1] = region_green[1, 1]
                result_image[i - 1 : i + 1, j - 1 : j + 1, 2] = region_blue[1, 1]

                # result_mask[i-1:i+1, j-1:j+1, 0] = 0
                # result_mask[i-1:i+1, j-1:j+1, 1] = 0
                # result_mask[i-1:i+1, j-1:j+1, 2] = 0

    return result_image  # , result_mask)
