from PIL import Image
from typing import Callable, List


def process_tiles(
    source: Image.Image,
    tile: int,
    scale: int,
    filters: List[Callable],
) -> Image:
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
            print('processing tile %s of %s, %s.%s' % (idx, total, y, x))
            tile_image = source.crop((left, top, left + tile, top + tile))

            for filter in filters:
                tile_image = filter(tile_image)

            image.paste(tile_image, (left * scale, top * scale))

    return image
