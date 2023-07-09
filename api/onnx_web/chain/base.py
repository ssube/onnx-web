from datetime import timedelta
from logging import getLogger
from time import monotonic
from typing import Any, List, Optional, Tuple

from PIL import Image

from ..output import save_image
from ..params import ImageParams, StageParams
from ..server import ServerContext
from ..utils import is_debug
from ..worker import ProgressCallback, WorkerContext
from .stage import BaseStage
from .tile import needs_tile, process_tile_order

logger = getLogger(__name__)


PipelineStage = Tuple[BaseStage, StageParams, Optional[dict]]


class ChainProgress:
    def __init__(self, parent: ProgressCallback, start=0) -> None:
        self.parent = parent
        self.step = start
        self.total = 0

    def __call__(self, step: int, timestep: int, latents: Any) -> None:
        if step < self.step:
            # accumulate on resets
            self.total += self.step

        self.step = step
        self.parent(self.get_total(), timestep, latents)

    def get_total(self) -> int:
        return self.step + self.total

    @classmethod
    def from_progress(cls, parent: ProgressCallback):
        start = parent.step if hasattr(parent, "step") else 0
        return ChainProgress(parent, start=start)


class ChainPipeline:
    """
    Run many stages in series, passing the image results from each to the next, and processing
    tiles as needed.
    """

    def __init__(
        self,
        stages: Optional[List[PipelineStage]] = None,
    ):
        """
        Create a new pipeline that will run the given stages.
        """
        self.stages = list(stages or [])

    def append(self, stage: Optional[PipelineStage]):
        """
        Append an additional stage to this pipeline.

        This requires an already-assembled `PipelineStage`. Use `ChainPipeline.stage` if you want the pipeline to
        assemble the stage from loose arguments.
        """
        if stage is not None:
            self.stages.append(stage)

    def run(
        self,
        job: WorkerContext,
        server: ServerContext,
        params: ImageParams,
        sources: List[Image.Image],
        callback: Optional[ProgressCallback],
        **kwargs
    ) -> List[Image.Image]:
        return self(job, server, params, sources=sources, callback=callback, **kwargs)

    def stage(self, callback: BaseStage, params: StageParams, **kwargs):
        self.stages.append((callback, params, kwargs))
        return self

    def __call__(
        self,
        job: WorkerContext,
        server: ServerContext,
        params: ImageParams,
        sources: List[Image.Image],
        callback: Optional[ProgressCallback] = None,
        **pipeline_kwargs
    ) -> List[Image.Image]:
        """
        DEPRECATED: use `run` instead
        """
        if callback is not None:
            callback = ChainProgress.from_progress(callback)

        start = monotonic()

        if len(sources) > 0:
            logger.info(
                "running pipeline on %s source images",
                len(sources),
            )
        else:
            sources = [None]
            logger.info("running pipeline without source images")

        stage_sources = sources
        for stage_pipe, stage_params, stage_kwargs in self.stages:
            name = stage_params.name or stage_pipe.__class__.__name__
            kwargs = stage_kwargs or {}
            kwargs = {**pipeline_kwargs, **kwargs}
            logger.debug(
                "running stage %s with %s source images, parameters: %s",
                name,
                len(stage_sources) - stage_sources.count(None),
                kwargs.keys(),
            )

            # the stage must be split and tiled if any image is larger than the selected/max tile size
            must_tile = any(
                [
                    needs_tile(
                        stage_pipe.max_tile,
                        stage_params.tile_size,
                        size=kwargs.get("size", None),
                        source=source,
                    )
                    for source in stage_sources
                ]
            )

            tile = stage_params.tile_size
            if stage_pipe.max_tile > 0:
                tile = min(stage_pipe.max_tile, stage_params.tile_size)

            if must_tile:
                stage_outputs = []
                for source in stage_sources:
                    logger.info(
                        "image larger than tile size of %s, tiling stage",
                        tile,
                    )

                    def stage_tile(source_tile: Image.Image, _dims) -> Image.Image:
                        output_tile = stage_pipe.run(
                            job,
                            server,
                            stage_params,
                            params,
                            [source_tile],
                            callback=callback,
                            **kwargs,
                        )[0]

                        if is_debug():
                            save_image(server, "last-tile.png", output_tile)

                        return output_tile

                    output = process_tile_order(
                        stage_params.tile_order,
                        source,
                        tile,
                        stage_params.outscale,
                        [stage_tile],
                        **kwargs,
                    )
                    stage_outputs.append(output)

                stage_sources = stage_outputs
            else:
                logger.debug("image within tile size of %s, running stage", tile)
                stage_sources = stage_pipe.run(
                    job,
                    server,
                    stage_params,
                    params,
                    stage_sources,
                    callback=callback,
                    **kwargs,
                )

            logger.debug(
                "finished stage %s with %s results",
                name,
                len(stage_sources),
            )

            if is_debug():
                save_image(server, "last-stage.png", stage_sources[0])

        end = monotonic()
        duration = timedelta(seconds=(end - start))
        logger.info(
            "finished pipeline in %s with %s results",
            duration,
            len(stage_sources),
        )
        return stage_sources
