from numpy import random
from PIL import Image, ImageFilter
from typing import Tuple

import numpy as np


def blend_mult(a):
    return float(a) / 256


def blend_imult(a):
    return 1.0 - blend_mult(a)


def blend_mask_source(source: Tuple[int, int, int], mask: Tuple[int, int, int], noise: Tuple[int, int, int]) -> Tuple[int, int, int]:
    '''
    Blend operation, linear interpolation from noise to source based on mask: `(s * (1 - m)) + (n * m)`
    Black = noise
    White = source
    '''
    return (
        int((source[0] * blend_mult(mask[0])) +
            (noise[0] * blend_imult(mask[0]))),
        int((source[1] * blend_mult(mask[1])) +
            (noise[1] * blend_imult(mask[1]))),
        int((source[2] * blend_mult(mask[2])) +
            (noise[2] * blend_imult(mask[2]))),
    )

def blend_source_mask(source: Tuple[int, int, int], mask: Tuple[int, int, int], noise: Tuple[int, int, int]) -> Tuple[int, int, int]:
    '''
    Blend operation, linear interpolation from source to noise based on mask: `(s * (1 - m)) + (n * m)`
    Black = source
    White = noise
    '''
    return (
        int((source[0] * blend_imult(mask[0])) +
            (noise[0] * blend_mult(mask[0]))),
        int((source[1] * blend_imult(mask[1])) +
            (noise[1] * blend_mult(mask[1]))),
        int((source[2] * blend_imult(mask[2])) +
            (noise[2] * blend_mult(mask[2]))),
    )


def noise_source_fill(source_image: Image, dims: Tuple[int, int], origin: Tuple[int, int], fill='white') -> Tuple[float, float, float]:
    '''
    Identity transform, source image centered on white canvas.
    '''
    width, height = dims

    noise = Image.new('RGB', (width, height), fill)
    noise.paste(source_image, origin)

    return noise


def noise_source_gaussian(source_image: Image, dims: Tuple[int, int], origin: Tuple[int, int], rounds=3) -> Tuple[float, float, float]:
    '''
    Gaussian blur, source image centered on white canvas.
    '''
    width, height = dims

    noise = Image.new('RGB', (width, height), 'white')
    noise.paste(source_image, origin)

    for i in range(rounds):
        noise.filter(ImageFilter.GaussianBlur(5))

    return noise


def noise_source_uniform(source_image: Image, dims: Tuple[int, int]) -> Tuple[float, float, float]:
    width, height = dims
    size = width * height

    noise_r = random.uniform(0, 256, size=size)
    noise_g = random.uniform(0, 256, size=size)
    noise_b = random.uniform(0, 256, size=size)

    noise = Image.new('RGB', (width, height))

    for x in range(width):
        for y in range(height):
            i = x * y
            noise.putpixel((x, y), (
                int(noise_r[i]),
                int(noise_g[i]),
                int(noise_b[i])
            ))

    return noise


def noise_source_normal(source_image: Image, dims: Tuple[int, int]) -> Tuple[float, float, float]:
    width, height = dims
    size = width * height

    noise_r = random.normal(128, 32, size=size)
    noise_g = random.normal(128, 32, size=size)
    noise_b = random.normal(128, 32, size=size)

    noise = Image.new('RGB', (width, height))

    for x in range(width):
        for y in range(height):
            i = x * y
            noise.putpixel((x, y), (
                int(noise_r[i]),
                int(noise_g[i]),
                int(noise_b[i])
            ))

    return noise


def noise_source_histogram(source_image: Image, dims: Tuple[int, int]) -> Tuple[float, float, float]:
    r, g, b = source_image.split()
    width, height = dims
    size = width * height

    hist_r = r.histogram()
    hist_g = g.histogram()
    hist_b = b.histogram()

    noise_r = random.choice(256, p=np.divide(
        np.copy(hist_r), np.sum(hist_r)), size=size)
    noise_g = random.choice(256, p=np.divide(
        np.copy(hist_g), np.sum(hist_g)), size=size)
    noise_b = random.choice(256, p=np.divide(
        np.copy(hist_b), np.sum(hist_b)), size=size)

    noise = Image.new('RGB', (width, height))

    for x in range(width):
        for y in range(height):
            i = x * y
            noise.putpixel((x, y), (
                noise_r[i],
                noise_g[i],
                noise_b[i]
            ))

    return noise


# based on https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/master/scripts/outpainting_mk_2.py#L175-L232
def expand_image(
        source_image: Image,
        mask_image: Image,
        expand_by: Tuple[int, int, int, int],
        fill='white',
        noise_source=noise_source_histogram,
        blend_op=blend_source_mask
):
    left, right, top, bottom = expand_by

    full_width = left + source_image.width + right
    full_height = top + source_image.height + bottom

    full_source = Image.new('RGB', (full_width, full_height), fill)
    full_source.paste(source_image, (left, top))

    full_mask = Image.new('RGB', (full_width, full_height), fill)
    full_mask.paste(mask_image, (left, top))

    full_noise = noise_source(source_image, (full_width, full_height), (top, left))

    for x in range(full_source.width):
        for y in range(full_source.height):
            mask_color = full_mask.getpixel((x, y))
            noise_color = full_noise.getpixel((x, y))
            source_color = full_source.getpixel((x, y))

            if mask_color[0] > 0:
                full_source.putpixel((x, y), blend_op(
                    source_color, mask_color, noise_color))

    return (full_source, full_mask, full_noise, (full_width, full_height))
