from typing import Tuple, Union

from PIL import Image, ImageChops, ImageOps

from ..params import Border, Size
from .mask_filter import mask_filter_none
from .noise_source import noise_source_histogram


# very loosely based on https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/master/scripts/outpainting_mk_2.py#L175-L232
def expand_image(
    source: Image.Image,
    mask: Image.Image,
    expand: Border,
    fill="white",
    noise_source=noise_source_histogram,
    mask_filter=mask_filter_none,
):
    size = Size(*source.size).add_border(expand).round_to_tile()
    size = tuple(size)
    origin = (expand.left, expand.top)

    full_source = Image.new("RGB", size, fill)
    full_source.paste(source, origin)

    # new mask pixels need to be filled with white so they will be replaced
    full_mask = mask_filter(mask, size, origin, fill="white")
    full_noise = noise_source(source, size, origin, fill=fill)
    full_noise = ImageChops.multiply(full_noise, full_mask)

    full_source = Image.composite(full_noise, full_source, full_mask.convert("L"))

    return (full_source, full_mask, full_noise, size)
