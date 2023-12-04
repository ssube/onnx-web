import numpy as np
from numpy import random
from PIL import Image, ImageFilter

from ..params import Point


def get_pixel_index(x: int, y: int, width: int) -> int:
    return (y * width) + x


def noise_source_fill_edge(
    source: Image.Image, dims: Point, origin: Point, fill="white", **kw
) -> Image.Image:
    """
    Identity transform, source image centered on white canvas.
    """
    width, height = dims

    noise = Image.new(source.mode, (width, height), fill)
    noise.paste(source, origin)

    return noise


def noise_source_fill_mask(
    source: Image.Image, dims: Point, _origin: Point, fill="white", **kw
) -> Image.Image:
    """
    Fill the whole canvas, no source or noise.
    """
    width, height = dims

    noise = Image.new(source.mode, (width, height), fill)

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
    source: Image.Image, dims: Point, _origin: Point, **kw
) -> Image.Image:
    width, height = dims
    size = width * height

    noise_r = random.uniform(0, 256, size=size)
    noise_g = random.uniform(0, 256, size=size)
    noise_b = random.uniform(0, 256, size=size)

    # needs to be RGB for pixel manipulation
    noise = Image.new("RGB", (width, height))

    for x in range(width):
        for y in range(height):
            i = get_pixel_index(x, y, width)
            noise.putpixel((x, y), (int(noise_r[i]), int(noise_g[i]), int(noise_b[i])))

    return noise.convert(source.mode)


def noise_source_normal(
    source: Image.Image, dims: Point, _origin: Point, **kw
) -> Image.Image:
    width, height = dims
    size = width * height

    noise_r = random.normal(128, 32, size=size)
    noise_g = random.normal(128, 32, size=size)
    noise_b = random.normal(128, 32, size=size)

    # needs to be RGB for pixel manipulation
    noise = Image.new("RGB", (width, height))

    for x in range(width):
        for y in range(height):
            i = get_pixel_index(x, y, width)
            noise.putpixel((x, y), (int(noise_r[i]), int(noise_g[i]), int(noise_b[i])))

    return noise.convert(source.mode)


def noise_source_histogram(
    source: Image.Image, dims: Point, _origin: Point, **kw
) -> Image.Image:
    r, g, b, *_a = source.split()
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

    # needs to be RGB for pixel manipulation
    noise = Image.new("RGB", (width, height))

    for x in range(width):
        for y in range(height):
            i = get_pixel_index(x, y, width)
            noise.putpixel((x, y), (noise_r[i], noise_g[i], noise_b[i]))

    return noise.convert(source.mode)
