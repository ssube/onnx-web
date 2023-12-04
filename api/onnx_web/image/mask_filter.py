from PIL import Image, ImageChops, ImageFilter

from ..params import Point


def mask_filter_none(
    mask: Image.Image, dims: Point, origin: Point, fill="white", **kw
) -> Image.Image:
    width, height = dims

    noise = Image.new(mask.mode, (width, height), fill)
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
