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
    origin = (expand.left, expand.top)

    full_source = Image.new("RGB", size, fill)
    full_source.paste(source, origin)

    # new mask pixels need to be filled with white so they will be replaced
    full_mask = mask_filter(mask, size, origin, fill="white")
    full_noise = noise_source(source, size, origin, fill=fill)
    full_noise = ImageChops.multiply(full_noise, full_mask)

    full_source = Image.composite(full_noise, full_source, full_mask.convert("L"))

    return (full_source, full_mask, full_noise, (size.width, size.height))


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
