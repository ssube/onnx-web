from logging import getLogger
from math import ceil
from typing import List, Protocol, Tuple

from PIL import Image

from ..params import TileOrder

logger = getLogger(__name__)


class TileCallback(Protocol):
    """
    Definition for a tile job function.
    """

    def __call__(self, image: Image.Image, dims: Tuple[int, int, int]) -> Image.Image:
        """
        Run this stage against a single tile.
        """
        pass


def process_tile_grid(
    source: Image.Image,
    tile: int,
    scale: int,
    filters: List[TileCallback],
    **kwargs,
) -> Image.Image:
    width, height = source.size
    image = Image.new("RGB", (width * scale, height * scale))

    tiles_x = width // tile
    tiles_y = height // tile
    total = tiles_x * tiles_y

    for y in range(tiles_y):
        for x in range(tiles_x):
            idx = (y * tiles_x) + x
            left = x * tile
            top = y * tile
            logger.debug("processing tile %s of %s, %s.%s", idx + 1, total, y, x)
            tile_image = source.crop((left, top, left + tile, top + tile))

            for filter in filters:
                tile_image = filter(tile_image, (left, top, tile))

            image.paste(tile_image, (left * scale, top * scale))

    return image


def process_tile_spiral(
    source: Image.Image,
    tile: int,
    scale: int,
    filters: List[TileCallback],
    overlap: float = 0.5,
    **kwargs,
) -> Image.Image:
    if scale != 1:
        raise ValueError("unsupported scale")

    width, height = source.size
    image = Image.new("RGB", (width * scale, height * scale))
    image.paste(source, (0, 0, width, height))

    # tile tuples is source, multiply by scale for dest
    counter = 0
    tiles = generate_tile_spiral(width, height, tile, overlap=overlap)
    for left, top in tiles:
        counter += 1
        logger.debug("processing tile %s of %s, %sx%s", counter, len(tiles), left, top)

        tile_image = image.crop((left, top, left + tile, top + tile))

        for filter in filters:
            tile_image = filter(tile_image, (left, top, tile))

        image.paste(tile_image, (left * scale, top * scale))

    return image


def process_tile_order(
    order: TileOrder,
    source: Image.Image,
    tile: int,
    scale: int,
    filters: List[TileCallback],
    **kwargs,
) -> Image.Image:
    if order == TileOrder.grid:
        logger.debug("using grid tile order with tile size: %s", tile)
        return process_tile_grid(source, tile, scale, filters, **kwargs)
    elif order == TileOrder.kernel:
        logger.debug("using kernel tile order with tile size: %s", tile)
        raise NotImplementedError()
    elif order == TileOrder.spiral:
        logger.debug("using spiral tile order with tile size: %s", tile)
        return process_tile_spiral(source, tile, scale, filters, **kwargs)
    else:
        logger.warn("unknown tile order: %s", order)
        raise ValueError()


def generate_tile_spiral(
    width: int,
    height: int,
    tile: int,
    overlap: float = 0.0,
) -> List[Tuple[int, int]]:
    spacing = 1.0 - overlap

    # round dims up to nearest tiles
    tile_width = ceil(width / tile)
    tile_height = ceil(height / tile)

    # start walking from the north-west corner, heading east
    dir_height = 0
    dir_width = 1

    walk_height = tile_height
    walk_width = tile_width

    accum_height = 0
    accum_width = 0

    tile_top = 0
    tile_left = 0

    tile_coords = []
    while walk_width > 0 and walk_height > 0:
        # exhaust the current direction, then turn
        while accum_width < walk_width and accum_height < walk_height:
            # add a tile
            # logger.trace(
            print(
                "adding tile at %s:%s, %s:%s, %s:%s",
                tile_left,
                tile_top,
                accum_width,
                accum_height,
                walk_width,
                walk_height,
                spacing,
            )
            tile_coords.append((tile_left, tile_top))

            # move to the next
            tile_top += dir_height * spacing * tile
            tile_left += dir_width * spacing * tile

            accum_height += abs(dir_height * spacing)
            accum_width += abs(dir_width * spacing)

        # reset for the next direction
        accum_height = 0
        accum_width = 0

        # why tho
        tile_top -= dir_height
        tile_left -= dir_width

        # turn right
        if dir_width == 1 and dir_height == 0:
            dir_width = 0
            dir_height = 1
        elif dir_width == 0 and dir_height == 1:
            dir_width = -1
            dir_height = 0
        elif dir_width == -1 and dir_height == 0:
            dir_width = 0
            dir_height = -1
        elif dir_width == 0 and dir_height == -1:
            dir_width = 1
            dir_height = 0

        # step to the next tile as part of the turn
        tile_top += dir_height
        tile_left += dir_width

        # shrink the last direction
        walk_height -= abs(dir_height)
        walk_width -= abs(dir_width)

    return tile_coords
