from typing import Tuple, Union

import numpy as np
from numpy import random
from PIL import Image, ImageChops, ImageFilter, ImageOps

from .params import Border, Point, Size


def get_pixel_index(x: int, y: int, width: int) -> int:
    return (y * width) + x


def mask_filter_none(
    mask: Image.Image, dims: Point, origin: Point, fill="white", **kw
) -> Image.Image:
    width, height = dims

    noise = Image.new("RGB", (width, height), fill)
    noise.paste(mask, origin)

    return noise


def mask_filter_gaussian_multiply(
    mask: Image.Image, dims: Point, origin: Point, rounds=3, **kw
) -> Image.Image:
    """
    Gaussian blur with multiply, source image centered on white canvas.
    """
    noise = mask_filter_none(mask, dims, origin)

    for _i in range(rounds):
        blur = noise.filter(ImageFilter.GaussianBlur(5))
        noise = ImageChops.multiply(noise, blur)

    return noise


def mask_filter_gaussian_screen(
    mask: Image.Image, dims: Point, origin: Point, rounds=3, **kw
) -> Image.Image:
    """
    Gaussian blur, source image centered on white canvas.
    """
    noise = mask_filter_none(mask, dims, origin)

    for _i in range(rounds):
        blur = noise.filter(ImageFilter.GaussianBlur(5))
        noise = ImageChops.screen(noise, blur)

    return noise


def noise_source_fill_edge(
    source: Image.Image, dims: Point, origin: Point, fill="white", **kw
) -> Image.Image:
    """
    Identity transform, source image centered on white canvas.
    """
    width, height = dims

    noise = Image.new("RGB", (width, height), fill)
    noise.paste(source, origin)

    return noise


def noise_source_fill_mask(
    _source: Image.Image, dims: Point, _origin: Point, fill="white", **kw
) -> Image.Image:
    """
    Fill the whole canvas, no source or noise.
    """
    width, height = dims

    noise = Image.new("RGB", (width, height), fill)

    return noise


def noise_source_gaussian(
    source: Image.Image, dims: Point, origin: Point, rounds=3, **kw
) -> Image.Image:
    """
    Gaussian blur, source image centered on white canvas.
    """
    noise = noise_source_uniform(source, dims, origin)
    noise.paste(source, origin)

    for _i in range(rounds):
        noise = noise.filter(ImageFilter.GaussianBlur(5))

    return noise


def noise_source_uniform(
    _source: Image.Image, dims: Point, _origin: Point, **kw
) -> Image.Image:
    width, height = dims
    size = width * height

    noise_r = random.uniform(0, 256, size=size)
    noise_g = random.uniform(0, 256, size=size)
    noise_b = random.uniform(0, 256, size=size)

    noise = Image.new("RGB", (width, height))

    for x in range(width):
        for y in range(height):
            i = get_pixel_index(x, y, width)
            noise.putpixel((x, y), (int(noise_r[i]), int(noise_g[i]), int(noise_b[i])))

    return noise


def noise_source_normal(
    _source: Image.Image, dims: Point, _origin: Point, **kw
) -> Image.Image:
    width, height = dims
    size = width * height

    noise_r = random.normal(128, 32, size=size)
    noise_g = random.normal(128, 32, size=size)
    noise_b = random.normal(128, 32, size=size)

    noise = Image.new("RGB", (width, height))

    for x in range(width):
        for y in range(height):
            i = get_pixel_index(x, y, width)
            noise.putpixel((x, y), (int(noise_r[i]), int(noise_g[i]), int(noise_b[i])))

    return noise


def noise_source_histogram(
    source: Image.Image, dims: Point, _origin: Point, **kw
) -> Image.Image:
    r, g, b = source.split()
    width, height = dims
    size = width * height

    hist_r = r.histogram()
    hist_g = g.histogram()
    hist_b = b.histogram()

    noise_r = random.choice(
        256, p=np.divide(np.copy(hist_r), np.sum(hist_r)), size=size
    )
    noise_g = random.choice(
        256, p=np.divide(np.copy(hist_g), np.sum(hist_g)), size=size
    )
    noise_b = random.choice(
        256, p=np.divide(np.copy(hist_b), np.sum(hist_b)), size=size
    )

    noise = Image.new("RGB", (width, height))

    for x in range(width):
        for y in range(height):
            i = get_pixel_index(x, y, width)
            noise.putpixel((x, y), (noise_r[i], noise_g[i], noise_b[i]))

    return noise


# very loosely based on https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/master/scripts/outpainting_mk_2.py#L175-L232
def expand_image(
    source: Image.Image,
    mask: Image.Image,
    expand: Border,
    fill="white",
    noise_source=noise_source_histogram,
    mask_filter=mask_filter_none,
):
    full_width = expand.left + source.width + expand.right
    full_height = expand.top + source.height + expand.bottom

    dims = (full_width, full_height)
    origin = (expand.left, expand.top)

    full_source = Image.new("RGB", dims, fill)
    full_source.paste(source, origin)

    # new mask pixels need to be filled with white so they will be replaced
    full_mask = mask_filter(mask, dims, origin, fill="white")
    full_noise = noise_source(source, dims, origin, fill=fill)
    full_noise = ImageChops.multiply(full_noise, full_mask)

    full_source = Image.composite(full_noise, full_source, full_mask.convert("L"))

    return (full_source, full_mask, full_noise, (full_width, full_height))


def valid_image(
    image: Image.Image,
    min_dims: Union[Size, Tuple[int, int]] = [512, 512],
    max_dims: Union[Size, Tuple[int, int]] = [512, 512],
) -> Image.Image:
    min_x, min_y = min_dims
    max_x, max_y = max_dims

    if image.width > max_x or image.height > max_y:
        image = ImageOps.contain(image, (max_x, max_y))

    if image.width < min_x or image.height < min_y:
        blank = Image.new(image.mode, (min_x, min_y), "black")
        blank.paste(image)
        image = blank

    # check for square

    return image
