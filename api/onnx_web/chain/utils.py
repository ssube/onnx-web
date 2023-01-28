from logging import getLogger
from PIL import Image
from typing import List, Protocol, Tuple

logger = getLogger(__name__)


class TileCallback(Protocol):
    def __call__(self, image: Image.Image, dims: Tuple[int, int, int]) -> Image.Image:
        pass


def process_tiles(
    source: Image.Image,
    tile: int,
    scale: int,
    filters: List[TileCallback],
) -> Image.Image:
    width, height = source.size
    image = Image.new('RGB', (width * scale, height * scale))

    tiles_x = width // tile
    tiles_y = height // tile
    total = tiles_x * tiles_y

    for y in range(tiles_y):
        for x in range(tiles_x):
            idx = (y * tiles_x) + x
            left = x * tile
            top = y * tile
            logger.info('processing tile %s of %s, %s.%s', idx + 1, total, y, x)
            tile_image = source.crop((left, top, left + tile, top + tile))

            for filter in filters:
                tile_image = filter(tile_image, (left, top, tile))

            image.paste(tile_image, (left * scale, top * scale))

    return image
