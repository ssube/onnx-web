from datetime import timedelta
from logging import getLogger
from time import monotonic
from typing import Any, List, Optional, Protocol, Tuple

from PIL import Image

from ..device_pool import JobContext
from ..output import save_image
from ..params import ImageParams, StageParams
from ..utils import ServerContext, is_debug
from .utils import process_tile_grid

logger = getLogger(__name__)


class StageCallback(Protocol):
    def __call__(
        self,
        job: JobContext,
        ctx: ServerContext,
        stage: StageParams,
        params: ImageParams,
        source: Image.Image,
        **kwargs: Any
    ) -> Image.Image:
        pass


PipelineStage = Tuple[StageCallback, StageParams, Optional[dict]]


class ChainPipeline:
    """
    Run many stages in series, passing the image results from each to the next, and processing
    tiles as needed.
    """

    def __init__(
        self,
        stages: List[PipelineStage] = [],
    ):
        """
        Create a new pipeline that will run the given stages.
        """
        self.stages = list(stages)

    def append(self, stage: PipelineStage):
        """
        Append an additional stage to this pipeline.
        """
        self.stages.append(stage)

    def __call__(
        self,
        job: JobContext,
        server: ServerContext,
        params: ImageParams,
        source: Image.Image,
        **pipeline_kwargs
    ) -> Image.Image:
        """
        TODO: handle List[Image] outputs
        """
        start = monotonic()
        logger.info(
            "running pipeline on source image with dimensions %sx%s",
            source.width,
            source.height,
        )
        image = source

        for stage_pipe, stage_params, stage_kwargs in self.stages:
            name = stage_params.name or stage_pipe.__name__
            kwargs = stage_kwargs or {}
            kwargs = {**pipeline_kwargs, **kwargs}

            logger.info(
                "running stage %s on image with dimensions %sx%s, %s",
                name,
                image.width,
                image.height,
                kwargs.keys(),
            )

            if (
                image.width > stage_params.tile_size
                or image.height > stage_params.tile_size
            ):
                logger.info(
                    "image larger than tile size of %s, tiling stage",
                    stage_params.tile_size,
                )

                def stage_tile(tile: Image.Image, _dims) -> Image.Image:
                    tile = stage_pipe(job, server, stage_params, params, tile, **kwargs)

                    if is_debug():
                        save_image(server, "last-tile.png", tile)

                    return tile

                image = process_tile_grid(
                    image, stage_params.tile_size, stage_params.outscale, [stage_tile]
                )
            else:
                logger.info("image within tile size, running stage")
                image = stage_pipe(job, server, stage_params, params, image, **kwargs)

            logger.info(
                "finished stage %s, result size: %sx%s", name, image.width, image.height
            )

            if is_debug():
                save_image(server, "last-stage.png", image)

        end = monotonic()
        duration = timedelta(seconds=(end - start))
        logger.info(
            "finished pipeline in %s, result size: %sx%s",
            duration,
            image.width,
            image.height,
        )
        return image
