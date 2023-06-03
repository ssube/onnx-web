from logging import getLogger
from math import ceil
from typing import List, Protocol, Tuple

import numpy as np
from PIL import Image
from skimage.exposure import match_histograms

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


def complete_tile(
    source: Image.Image,
    tile: int,
) -> Image.Image:
    if source.width < tile or source.height < tile:
        full_source = Image.new(source.mode, (tile, tile))
        full_source.paste(source)
        return full_source

    return source


def get_tile_grads(
    left: int,
    top: int,
    tile: int,
    width: int,
    height: int,
) -> Tuple[Tuple[float, float, float, float], Tuple[float, float, float, float]]:
    grad_x = [0, 1, 1, 0]
    grad_y = [0, 1, 1, 0]

    if left <= 0:
        grad_x[0] = 1

    if top <= 0:
        grad_y[0] = 1

    if (left + tile) >= width:
        grad_x[3] = 1

    if (top + tile) >= height:
        grad_y[3] = 1

    return (grad_x, grad_y)


def process_tile_grid(
    source: Image.Image,
    tile: int,
    scale: int,
    filters: List[TileCallback],
    overlap: float = 0.5,
    **kwargs,
) -> Image.Image:
    width, height = source.size

    adj_tile = int(float(tile) * overlap)
    tiles_x = ceil(width / adj_tile)
    tiles_y = ceil(height / adj_tile)
    total = tiles_x * tiles_y

    tiles: List[Tuple[int, int, Image.Image]] = []

    for y in range(tiles_y):
        for x in range(tiles_x):
            idx = (y * tiles_x) + x
            left = x * adj_tile
            top = y * adj_tile
            logger.debug("processing tile %s of %s, %s.%s", idx + 1, total, y, x)

            tile_image = source.crop((left, top, left + tile, top + tile))
            tile_image = complete_tile(tile_image, tile)

            for filter in filters:
                tile_image = filter(tile_image, (left, top, tile))

            tiles.append((left, top, tile_image))

    scaled_size = (height * scale, width * scale, 3)
    count = np.zeros(scaled_size)
    value = np.zeros(scaled_size)
    ref = np.array(tiles[0][2])

    for left, top, tile_image in tiles:
        # histogram equalization
        equalized = np.array(tile_image)
        equalized = match_histograms(equalized, ref, channel_axis=-1)

        # gradient blending
        points = [0, adj_tile * scale, (tile - adj_tile) * scale, (tile * scale) - 1]
        grad_x, grad_y = get_tile_grads(left, top, adj_tile, width, height)
        mult_x = [np.interp(i, points, grad_x) for i in range(tile * scale)]
        mult_y = [np.interp(i, points, grad_y) for i in range(tile * scale)]

        mask = np.ones_like(equalized[:, :, 0]) * mult_x
        mask = (mask.T * mult_y).T
        for c in range(3):
            equalized[:, :, c] = (equalized[:, :, c] * mask).astype(np.uint8)

        # accumulation
        # equalized size may be wrong/too much
        scaled_top = top * scale
        scaled_left = left * scale

        scaled_bottom = min(scaled_top + equalized.shape[0], scaled_size[0])
        scaled_right = min(scaled_left + equalized.shape[1], scaled_size[1])

        value[
            scaled_top : scaled_bottom, scaled_left : scaled_right, :
        ] += equalized[0 : scaled_bottom - scaled_top, 0 : scaled_right - scaled_left, :]
        count[
            scaled_top : scaled_bottom, scaled_left : scaled_right, :
        ] += np.repeat(mask[0 : scaled_bottom - scaled_top, 0 : scaled_right - scaled_left, np.newaxis], 3, axis=2)

    pixels = np.where(count > 0, value / count, value)
    return Image.fromarray(pixels)


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
        tile_image = complete_tile(tile_image, tile)

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
            logger.trace(
                "adding tile at %s:%s, %s:%s, %s:%s, %s",
                tile_left,
                tile_top,
                accum_width,
                accum_height,
                walk_width,
                walk_height,
                spacing,
            )
            tile_coords.append((int(tile_left), int(tile_top)))

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
