import itertools
from enum import Enum
from logging import getLogger
from math import ceil
from typing import List, Optional, Protocol, Tuple

import numpy as np
from PIL import Image

from ..image.noise_source import noise_source_histogram
from ..params import Size, TileOrder

# from skimage.exposure import match_histograms


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
    if source is None:
        return source

    if source.width < tile or source.height < tile:
        full_source = Image.new(source.mode, (tile, tile))
        full_source.paste(source)
        return full_source

    return source


def needs_tile(
    max_tile: int,
    stage_tile: int,
    size: Optional[Size] = None,
    source: Optional[Image.Image] = None,
) -> bool:
    tile = min(max_tile, stage_tile)
    logger.trace(
        "checking image tile dimensions: %s, %s, %s",
        tile,
        source.width > tile or source.height > tile,
        size.width > tile or size.height > tile,
    )

    if source is not None:
        return source.width > tile or source.height > tile

    if size is not None:
        return size.width > tile or size.height > tile

    return False


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


def blend_tiles(
    tiles: List[Tuple[int, int, Image.Image]],
    scale: int,
    width: int,
    height: int,
    tile: int,
    overlap: float,
):
    adj_tile = int(float(tile) * (1.0 - overlap))
    logger.debug(
        "adjusting tile size from %s to %s based on %s overlap", tile, adj_tile, overlap
    )

    scaled_size = (height * scale, width * scale, 3)
    count = np.zeros(scaled_size)
    value = np.zeros(scaled_size)

    for left, top, tile_image in tiles:
        # histogram equalization
        equalized = np.array(tile_image).astype(np.float32)
        mask = np.ones_like(equalized[:, :, 0])

        if adj_tile < tile:
            # sort gradient points
            p1 = adj_tile * scale
            p2 = (tile - adj_tile) * scale
            points = [0, min(p1, p2), max(p1, p2), tile * scale]

            # gradient blending
            grad_x, grad_y = get_tile_grads(left, top, adj_tile, width, height)
            logger.debug("tile gradients: %s, %s, %s", points, grad_x, grad_y)

            mult_x = [np.interp(i, points, grad_x) for i in range(tile * scale)]
            mult_y = [np.interp(i, points, grad_y) for i in range(tile * scale)]

            mask = ((mask * mult_x).T * mult_y).T
            for c in range(3):
                equalized[:, :, c] = equalized[:, :, c] * mask

        scaled_top = top * scale
        scaled_left = left * scale

        # equalized size may be wrong/too much
        scaled_bottom = scaled_top + equalized.shape[0]
        scaled_right = scaled_left + equalized.shape[1]

        writable_top = max(scaled_top, 0)
        writable_left = max(scaled_left, 0)
        writable_bottom = min(scaled_bottom, scaled_size[0])
        writable_right = min(scaled_right, scaled_size[1])

        margin_top = writable_top - scaled_top
        margin_left = writable_left - scaled_left
        margin_bottom = writable_bottom - scaled_bottom
        margin_right = writable_right - scaled_right

        logger.debug(
            "tile broadcast shapes: %s, %s, %s, %s \n writing shapes: %s, %s, %s, %s",
            writable_top,
            writable_left,
            writable_bottom,
            writable_right,
            margin_top,
            equalized.shape[0] + margin_bottom,
            margin_left,
            equalized.shape[0] + margin_right,
        )

        # accumulation
        value[
            writable_top:writable_bottom, writable_left:writable_right, :
        ] += equalized[
            margin_top : equalized.shape[0] + margin_bottom,
            margin_left : equalized.shape[1] + margin_right,
            :,
        ]
        count[
            writable_top:writable_bottom, writable_left:writable_right, :
        ] += np.repeat(
            mask[
                margin_top : equalized.shape[0] + margin_bottom,
                margin_left : equalized.shape[1] + margin_right,
                np.newaxis,
            ],
            3,
            axis=2,
        )

    logger.trace("mean tiles contributing to each pixel: %s", np.mean(count))
    pixels = np.where(count > 0, value / count, value)
    return Image.fromarray(np.uint8(pixels))


def process_tile_grid(
    source: Image.Image,
    tile: int,
    scale: int,
    filters: List[TileCallback],
    overlap: float = 0.0,
    **kwargs,
) -> Image.Image:
    width, height = kwargs.get("size", source.size if source else None)

    adj_tile = int(float(tile) * (1.0 - overlap))
    tiles_x = ceil(width / adj_tile)
    tiles_y = ceil(height / adj_tile)
    total = tiles_x * tiles_y
    logger.debug(
        "processing %s tiles (%s x %s) with adjusted size of %s, %s overlap",
        total,
        tiles_x,
        tiles_y,
        adj_tile,
        overlap,
    )

    tiles: List[Tuple[int, int, Image.Image]] = []

    for y in range(tiles_y):
        for x in range(tiles_x):
            idx = (y * tiles_x) + x
            left = x * adj_tile
            top = y * adj_tile
            logger.info("processing tile %s of %s, %s.%s", idx + 1, total, y, x)

            tile_image = (
                source.crop((left, top, left + tile, top + tile)) if source else None
            )
            tile_image = complete_tile(tile_image, tile)

            for filter in filters:
                tile_image = filter(tile_image, (left, top, tile))

            tiles.append((left, top, tile_image))

    return blend_tiles(tiles, scale, width, height, tile, overlap)


def process_tile_spiral(
    source: Image.Image,
    tile: int,
    scale: int,
    filters: List[TileCallback],
    overlap: float = 0.5,
    **kwargs,
) -> Image.Image:
    width, height = kwargs.get("size", source.size if source else None)
    mask = kwargs.get("mask", None)
    noise_source = kwargs.get("noise_source", noise_source_histogram)
    fill_color = kwargs.get("fill_color", None)
    if not mask:
        tile_mask = None

    tiles: List[Tuple[int, int, Image.Image]] = []

    # tile tuples is source, multiply by scale for dest
    counter = 0
    tile_coords = generate_tile_spiral(width, height, tile, overlap=overlap)

    if len(tile_coords) == 1:
        single_tile = True
    else:
        single_tile = False

    for left, top in tile_coords:
        counter += 1
        logger.info(
            "processing tile %s of %s, %sx%s", counter, len(tile_coords), left, top
        )

        right = left + tile
        bottom = top + tile

        left_margin = right_margin = top_margin = bottom_margin = 0
        needs_margin = False

        if left < 0:
            needs_margin = True
            left_margin = 0 - left
        if right > width:
            needs_margin = True
            right_margin = width - right
        if top < 0:
            needs_margin = True
            top_margin = 0 - top
        if bottom > height:
            needs_margin = True
            bottom_margin = height - bottom

        # if no source given, we don't have a source image
        if not source:
            tile_image = None
        elif needs_margin:
            # in the special case where the image is smaller than the specified tile size, just use the image
            if single_tile:
                logger.debug("creating and processing single-tile subtile")
                tile_image = source
                if mask:
                    tile_mask = mask
            # otherwise use add histogram noise outside of the image border
            else:
                logger.debug(
                    "tiling and adding margins: %s, %s, %s, %s",
                    left_margin,
                    top_margin,
                    right_margin,
                    bottom_margin,
                )
                base_image = source.crop(
                    (
                        left + left_margin,
                        top + top_margin,
                        right + right_margin,
                        bottom + bottom_margin,
                    )
                )
                tile_image = noise_source(
                    base_image, (tile, tile), (0, 0), fill=fill_color
                )
                tile_image.paste(base_image, (left_margin, top_margin))

                if mask:
                    base_mask = mask.crop(
                        (
                            left + left_margin,
                            top + top_margin,
                            right + right_margin,
                            bottom + bottom_margin,
                        )
                    )
                    tile_mask = Image.new("L", (tile, tile), color=0)
                    tile_mask.paste(base_mask, (left_margin, top_margin))

        else:
            logger.debug("tiling normally")
            tile_image = source.crop((left, top, right, bottom))
            if mask:
                tile_mask = mask.crop((left, top, right, bottom))

        for image_filter in filters:
            tile_image = image_filter(tile_image, tile_mask, (left, top, tile))

        tiles.append((left, top, tile_image))

    if single_tile:
        return tile_image
    else:
        return blend_tiles(tiles, scale, width, height, tile, overlap)


def process_tile_order(
    order: TileOrder,
    source: Image.Image,
    tile: int,
    scale: int,
    filters: List[TileCallback],
    **kwargs,
) -> Image.Image:
    """
    TODO: needs to handle more than one image
    """
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
        logger.warning("unknown tile order: %s", order)
        raise ValueError()


def generate_tile_spiral(
    width: int,
    height: int,
    tile: int,
    overlap: float = 0.0,
) -> List[Tuple[int, int]]:
    spacing = 1.0 - overlap

    tile_increment = (
        round(tile * spacing / 2) * 2
    )  # dividing and then multiplying by 2 ensures this will be an even number, which is necessary for the initial tile placement calculation

    # calculate the number of tiles needed
    if width > tile:
        width_tile_target = 1 + ceil((width - tile) / tile_increment)
    else:
        width_tile_target = 1
    if height > tile:
        height_tile_target = 1 + ceil((height - tile) / tile_increment)
    else:
        height_tile_target = 1

    # calculate the start position of the tiling
    span_x = tile + (width_tile_target - 1) * tile_increment
    span_y = tile + (height_tile_target - 1) * tile_increment

    logger.debug("tiled image overlap: %s. Span: %s x %s", overlap, span_x, span_y)

    tile_left = (
        width - span_x
    ) / 2  # guaranteed to be an integer because width and span will both be even
    tile_top = (
        height - span_y
    ) / 2  # guaranteed to be an integer because width and span will both be even

    logger.debug(
        "image size %s x %s, tiling to %s x %s, starting at %s, %s",
        width,
        height,
        width_tile_target,
        height_tile_target,
        tile_left,
        tile_top,
    )

    tile_coords = []

    # start walking from the north-west corner, heading east
    class WalkState(Enum):
        EAST = (1, 0)
        SOUTH = (0, 1)
        WEST = (-1, 0)
        NORTH = (0, -1)

    # initialize the tile_left placement
    tile_left -= tile_increment
    height_tile_target -= 1

    for state in itertools.cycle(WalkState):
        # This expression is stupid, but all it does is calculate the number of tiles we need in the appropriate direction
        accum_tile_target = max(
            map(
                lambda coord, val: abs(coord * val),
                state.value,
                (width_tile_target, height_tile_target),
            )
        )
        # check if done
        if accum_tile_target == 0:
            break

        # reset tile count
        accum_tiles = 0
        while accum_tiles < accum_tile_target:
            # move to the next
            tile_left += tile_increment * state.value[0]
            tile_top += tile_increment * state.value[1]

            # add a tile
            logger.debug("adding tile at %s:%s", tile_left, tile_top)
            tile_coords.append((int(tile_left), int(tile_top)))

            accum_tiles += 1

        width_tile_target -= abs(state.value[0])
        height_tile_target -= abs(state.value[1])

    return tile_coords
