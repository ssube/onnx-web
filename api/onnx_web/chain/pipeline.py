from datetime import timedelta
from logging import getLogger
from time import monotonic
from typing import Any, List, Optional, Tuple

from PIL import Image

from ..errors import CancelledException, RetryException
from ..output import save_image
from ..params import ImageParams, Size, StageParams
from ..server import ServerContext
from ..utils import is_debug, run_gc
from ..worker import ProgressCallback, WorkerContext
from ..worker.command import Progress
from .base import BaseStage
from .result import StageResult
from .tile import needs_tile, process_tile_order

logger = getLogger(__name__)


PipelineStage = Tuple[BaseStage, StageParams, Optional[dict]]


class ChainProgress:
    parent: ProgressCallback
    step: int  # same as steps.current, left for legacy purposes
    prev: int  # accumulator when step resets

    # new progress trackers
    steps: Progress
    stages: Progress
    tiles: Progress
    result: Optional[StageResult]

    def __init__(self, parent: ProgressCallback, start=0) -> None:
        self.parent = parent
        self.step = start
        self.prev = 0
        self.steps = Progress(self.step, self.prev)
        self.stages = Progress(0, 0)
        self.tiles = Progress(0, 0)
        self.result = None

    def __call__(self, step: int, timestep: int, latents: Any) -> None:
        if step < self.step:
            # accumulate on resets
            self.prev += self.step

        self.step = step

        total = self.get_total()
        self.steps.current = total
        self.parent(total, timestep, latents)

    def get_total(self) -> int:
        return self.step + self.prev

    def set_total(self, steps: int, stages: int = 0, tiles: int = 0) -> None:
        self.prev = steps
        self.steps.total = steps
        self.stages.total = stages
        self.tiles.total = tiles

    @classmethod
    def from_progress(cls, parent: ProgressCallback):
        start = parent.step if hasattr(parent, "step") else 0
        return ChainProgress(parent, start=start)


class ChainPipeline:
    """
    Run many stages in series, passing the image results from each to the next, and processing
    tiles as needed.
    """

    stages: List[PipelineStage]

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
        worker: WorkerContext,
        server: ServerContext,
        params: ImageParams,
        sources: StageResult,
        callback: Optional[ProgressCallback],
        **kwargs,
    ) -> List[Image.Image]:
        result = self(
            worker, server, params, sources=sources, callback=callback, **kwargs
        )
        return result.as_image()

    def stage(self, callback: BaseStage, params: StageParams, **kwargs):
        self.stages.append((callback, params, kwargs))
        return self

    def steps(self, params: ImageParams, size: Size) -> int:
        steps = 0
        for callback, _params, kwargs in self.stages:
            steps += callback.steps(kwargs.get("params", params), size)

        return steps

    def stages(self) -> int:
        return len(self.stages)

    def __call__(
        self,
        worker: WorkerContext,
        server: ServerContext,
        params: ImageParams,
        sources: StageResult,
        callback: Optional[ProgressCallback] = None,
        **pipeline_kwargs,
    ) -> StageResult:
        if callback is None:
            callback = worker.get_progress_callback()

        # wrap the progress counter in a one that can be reset if needed
        if not isinstance(callback, ChainProgress):
            callback = ChainProgress.from_progress(callback)

        # set estimated totals
        # TODO: should use self.steps, but size is not available here
        callback.set_total(params.steps, stages=len(self.stages), tiles=0)

        start = monotonic()

        if len(sources) > 0:
            logger.info(
                "running pipeline on %s source images",
                len(sources),
            )
        else:
            logger.info("running pipeline without source images")

        stage_sources = sources
        for stage_i, (stage_pipe, stage_params, stage_kwargs) in enumerate(self.stages):
            name = stage_params.name or stage_pipe.__class__.__name__
            kwargs = stage_kwargs or {}
            kwargs = {**pipeline_kwargs, **kwargs}
            logger.debug(
                "running stage %s with %s source images, parameters: %s",
                name,
                len(stage_sources),
                kwargs.keys(),
            )
            callback.stages.current = stage_i

            per_stage_params = params
            if "params" in kwargs:
                per_stage_params = kwargs["params"]
                kwargs.pop("params")

            # the stage must be split and tiled if any image is larger than the selected/max tile size
            must_tile = has_mask(stage_kwargs) or any(
                [
                    needs_tile(
                        stage_pipe.max_tile,
                        stage_params.tile_size,
                        size=kwargs.get("size", None),
                        source=source,
                    )
                    for source in stage_sources.as_image()
                ]
            )

            tile = stage_params.tile_size
            if stage_pipe.max_tile > 0:
                tile = min(stage_pipe.max_tile, stage_params.tile_size)

            callback.tiles.current = 0  # reset this either way
            if must_tile:
                logger.info(
                    "image contains sources or is larger than tile size of %s, tiling stage",
                    tile,
                )

                def stage_tile(
                    source_tile: List[Image.Image],
                    tile_mask: Image.Image,
                    dims: Tuple[int, int, int],
                ) -> List[Image.Image]:
                    for _i in range(worker.retries):
                        try:
                            tile_result = stage_pipe.run(
                                worker,
                                server,
                                stage_params,
                                per_stage_params,
                                StageResult(
                                    images=source_tile, metadata=stage_sources.metadata
                                ),
                                tile_mask=tile_mask,
                                callback=callback,
                                dims=dims,
                                **kwargs,
                            )

                            if is_debug():
                                for j, image in enumerate(tile_result.as_image()):
                                    save_image(server, f"last-tile-{j}.png", image)

                            callback.tiles.current = callback.tiles.current + 1

                            return tile_result
                        except CancelledException as err:
                            worker.retries = 0
                            logger.exception("job was cancelled while tiling")
                            raise err
                        except Exception:
                            worker.retries = worker.retries - 1
                            logger.exception(
                                "error while running stage pipeline for tile, %s retries left",
                                worker.retries,
                            )
                            server.cache.clear()
                            run_gc([worker.get_device()])

                    raise RetryException("exhausted retries on tile")

                stage_results = process_tile_order(
                    stage_params.tile_order,
                    stage_sources,
                    tile,
                    stage_params.outscale,
                    [stage_tile],
                    **kwargs,
                )

                stage_sources = StageResult(
                    images=stage_results, metadata=stage_sources.metadata
                )
            else:
                logger.debug(
                    "image does not contain sources and is within tile size of %s, running stage",
                    tile,
                )
                for _i in range(worker.retries):
                    try:
                        stage_result = stage_pipe.run(
                            worker,
                            server,
                            stage_params,
                            per_stage_params,
                            stage_sources,
                            callback=callback,
                            dims=(0, 0, tile),
                            **kwargs,
                        )
                        # doing this on the same line as stage_pipe.run can leave sources as None, which the pipeline
                        # does not like, so it throws
                        stage_sources = stage_result
                        break
                    except CancelledException as err:
                        worker.retries = 0
                        logger.exception("job was cancelled during stage")
                        raise err
                    except Exception:
                        worker.retries = worker.retries - 1
                        logger.exception(
                            "error while running stage pipeline, %s retries left",
                            worker.retries,
                        )
                        server.cache.clear()
                        run_gc([worker.get_device()])

                if worker.retries <= 0:
                    raise RetryException("exhausted retries on stage")

            logger.debug(
                "finished stage %s with %s results",
                name,
                len(stage_sources),
            )

            callback.result = (
                stage_sources  # this has just been set to the result of the last stage
            )

            if is_debug():
                for j, image in enumerate(stage_sources.as_image()):
                    save_image(server, f"last-stage-{j}.png", image)

        end = monotonic()
        duration = timedelta(seconds=(end - start))
        logger.info(
            "finished pipeline in %s with %s results",
            duration,
            len(stage_sources),
        )

        callback.result = stage_sources
        return stage_sources


MASK_KEYS = ["mask", "stage_mask", "tile_mask"]


def has_mask(args: List[str]) -> bool:
    return any([key in args for key in MASK_KEYS])
