from PIL import Image
from os import path
from typing import Any, List, Optional, Protocol, Tuple

from .image import (
    process_tiles,
)
from .utils import (
    ImageParams,
    ServerContext,
)


class StageParams:
    '''
    Parameters for a pipeline stage, assuming they can be chained.
    '''

    def __init__(
        self,
        tile_size: int = 512,
        outscale: int = 1,
    ) -> None:
        self.tile_size = tile_size
        self.outscale = outscale


class StageCallback(Protocol):
    def __call__(
        self,
        ctx: ServerContext,
        stage: StageParams,
        params: ImageParams,
        source: Image.Image,
        **kwargs: Any
    ) -> Image.Image:
        pass


PipelineStage = Tuple[StageCallback, StageParams, Optional[Any]]


class ChainPipeline:
    '''
    Run many stages in series, passing the image results from each to the next, and processing
    tiles as needed.
    '''

    def __init__(
        self,
        stages: List[PipelineStage],
    ):
        '''
        Create a new pipeline that will run the given stages.
        '''
        self.stages = stages

    def append(self, stage: PipelineStage):
        '''
        Append an additional stage to this pipeline.
        '''
        self.stages.append(stage)

    def __call__(self, ctx: ServerContext, params: ImageParams, source: Image.Image) -> Image.Image:
        '''
        TODO: handle List[Image] outputs
        '''
        print('running pipeline on source image with dimensions %sx%s' %
              source.size)
        image = source

        for stage_fn, stage_params, stage_kwargs in self.stages:
            print('running pipeline stage on result image with dimensions %sx%s' %
                  image.size)
            if image.width > stage_params.tile_size or image.height > stage_params.tile_size:
                print('source image larger than tile size, tiling stage',
                      stage_params.tile_size)

                def stage_tile(tile: Image.Image) -> Image.Image:
                    tile = stage_fn(ctx, stage_params, tile,
                                    params, **stage_kwargs)
                    tile.save(path.join(ctx.output_path, 'last-tile.png'))
                    return tile

                image = process_tiles(
                    image, stage_params.tile_size, stage_params.outscale, [stage_tile])
            else:
                print('source image within tile size, run stage')
                image = stage_fn(ctx, stage_params, image,
                                         params, **stage_kwargs)

            print('finished running pipeline stage, result size: %sx%s' % image.size)

        print('finished running pipeline, result size: %sx%s' % image.size)
        return image
