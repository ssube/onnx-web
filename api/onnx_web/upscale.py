from PIL import Image

from .chain import (
    correct_gfpgan,
    upscale_stable_diffusion,
    upscale_resrgan,
    ChainPipeline,
)
from .params import (
    ImageParams,
    StageParams,
    UpscaleParams,
)
from .utils import (
    ServerContext,
)


def run_upscale_correction(
    ctx: ServerContext,
    stage: StageParams,
    params: ImageParams,
    image: Image.Image,
    *,
    upscale: UpscaleParams,
) -> Image.Image:
    print('running upscale pipeline')

    chain = ChainPipeline()
    kwargs = {'upscale': upscale}

    if upscale.scale > 1:
        if 'esrgan' in upscale.upscale_model:
            stage = StageParams(tile_size=stage.tile_size,
                                outscale=upscale.outscale)
            chain.append((upscale_resrgan, stage, kwargs))
        elif 'stable-diffusion' in upscale.upscale_model:
            mini_tile = min(128, stage.tile_size)
            stage = StageParams(tile_size=mini_tile, outscale=upscale.outscale)
            chain.append((upscale_stable_diffusion, stage, kwargs))

    if upscale.faces:
        stage = StageParams(tile_size=stage.tile_size,
                            outscale=upscale.outscale)
        chain.append((correct_gfpgan, stage, kwargs))

    return chain(ctx, params, image)
