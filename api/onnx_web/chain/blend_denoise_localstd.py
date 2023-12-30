from logging import getLogger
from typing import Optional, Tuple

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
        strength: int = 5,
        range: int = 4,
        stage_source: Optional[Image.Image] = None,
        callback: Optional[ProgressCallback] = None,
        **kwargs,
    ) -> StageResult:
        logger.info("denoising source images")

        return StageResult.from_arrays(
            [
                remove_noise(source, threshold=strength, deviation=range)[0]
                for source in sources.as_numpy()
            ]
        )


def downscale_image(image: np.ndarray, scale: int = 2):
    result_image = np.zeros(
        (image.shape[0] // scale, image.shape[1] // scale), dtype=np.uint8
    )

    for i in range(0, image.shape[0] - scale, scale):
        for j in range(0, image.shape[1] - scale, scale):
            # Average the four neighboring pixels
            pixel_average = np.mean(image[i : i + scale, j : j + scale], axis=(0, 1))
            result_image[i // scale, j // scale] = pixel_average.astype(np.uint8)

    return result_image


def replace_noise(region: np.ndarray, threshold: int, deviation: float, op=np.median):
    # Identify stray pixels (brightness significantly deviates from surrounding pixels)
    central_pixel = np.mean(region[2:4, 2:4])

    region_normal = op(region)
    region_deviation = np.std(region)
    diff = np.abs(central_pixel - region_normal)

    # If the whole region is fairly consistent but the central pixel deviates significantly,
    if diff > (region_deviation + threshold) and diff < (
        region_deviation + threshold * deviation
    ):
        surrounding_pixels = region[region != central_pixel]
        surrounding_median = op(surrounding_pixels)
        # replace it with the median of surrounding pixels
        region[1, 1] = surrounding_median
        return True

    return False


def remove_noise(
    image: np.ndarray,
    threshold: int,
    deviation: float,
    region_size: Tuple[int, int] = (6, 6),
):
    # Create a copy of the original image to store the result
    result_image = np.copy(image)
    result_mask = np.zeros_like(image)

    # Iterate over regions in each channel
    i_inc = region_size[0] // 2
    j_inc = region_size[1] // 2

    for i in range(i_inc, image.shape[0] - i_inc, 1):
        for j in range(j_inc, image.shape[1] - j_inc, 1):
            i_min = i - (region_size[0] // 2)
            i_max = i + (region_size[0] // 2)
            j_min = j - (region_size[1] // 2)
            j_max = j + (region_size[1] // 2)
            # print(i_min, i_max, j_min, j_max)

            # skip if the central pixels have already been masked by a previous artifact
            if np.any(result_mask[i - 1 : i + 1, j - 1 : j + 1] > 0):
                continue

            # Extract region from each channel
            region_red = image[i_min:i_max, j_min:j_max, 0]
            region_green = image[i_min:i_max, j_min:j_max, 1]
            region_blue = image[i_min:i_max, j_min:j_max, 2]

            replaced = any(
                [
                    replace_noise(region_red, threshold, deviation),
                    replace_noise(region_green, threshold, deviation),
                ]
            )

            # apply the noise removal function to each channel
            if replaced:
                # assign the processed region back to the result image
                result_image[i - 1 : i + 1, j - 1 : j + 1, 0] = region_red[1, 1]
                result_image[i - 1 : i + 1, j - 1 : j + 1, 1] = region_green[1, 1]
                result_image[i - 1 : i + 1, j - 1 : j + 1, 2] = region_blue[1, 1]

                result_mask[i - 1 : i + 1, j - 1 : j + 1, 0] = 1
                result_mask[i - 1 : i + 1, j - 1 : j + 1, 1] = 1
                result_mask[i - 1 : i + 1, j - 1 : j + 1, 2] = 1

    return (result_image, result_mask * 255)
