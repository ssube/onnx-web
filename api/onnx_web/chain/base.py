from PIL import Image
from os import path
from typing import Any, List, Optional, Protocol, Tuple

from ..params import (
    ImageParams,
    StageParams,
)
from ..utils import (
    is_debug,
    ServerContext,
)
from .utils import (
    process_tiles,
)


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


PipelineStage = Tuple[StageCallback, StageParams, Optional[dict]]


class ChainPipeline:
    '''
    Run many stages in series, passing the image results from each to the next, and processing
    tiles as needed.
    '''

    def __init__(
        self,
        stages: List[PipelineStage] = [],
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

        for stage_pipe, stage_params, stage_kwargs in self.stages:
            name = stage_params.name or stage_pipe.__name__
            kwargs = stage_kwargs or {}
            print('running pipeline stage %s on result image with dimensions %sx%s' %
                  (name, image.width, image.height))

            if image.width > stage_params.tile_size or image.height > stage_params.tile_size:
                print('source image larger than tile size of %s, tiling stage' % (
                    stage_params.tile_size))

                def stage_tile(tile: Image.Image, _dims) -> Image.Image:
                    tile = stage_pipe(ctx, stage_params, params, tile,
                                      **kwargs)

                    if is_debug():
                        tile.save(path.join(ctx.output_path, 'last-tile.png'))

                    return tile

                image = process_tiles(
                    image, stage_params.tile_size, stage_params.outscale, [stage_tile])
            else:
                print('source image within tile size, running stage')
                image = stage_pipe(ctx, stage_params, params, image,
                                   **kwargs)

            print('finished running pipeline stage %s, result size: %sx%s' %
                  (name, image.width, image.height))

        print('finished running pipeline, result size: %sx%s' % image.size)
        return image
