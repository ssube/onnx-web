from datetime import timedelta
from logging import getLogger
from time import monotonic
from typing import Any, List, Optional, Tuple

from PIL import Image

from ..errors import RetryException
from ..output import save_image
from ..params import ImageParams, Size, StageParams
from ..server import ServerContext
from ..utils import is_debug, run_gc
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
        worker: WorkerContext,
        server: ServerContext,
        params: ImageParams,
        sources: List[Image.Image],
        callback: Optional[ProgressCallback],
        **kwargs
    ) -> List[Image.Image]:
        return self(
            worker, server, params, sources=sources, callback=callback, **kwargs
        )

    def stage(self, callback: BaseStage, params: StageParams, **kwargs):
        self.stages.append((callback, params, kwargs))
        return self

    def steps(self, params: ImageParams, size: Size):
        steps = 0
        for callback, _params, kwargs in self.stages:
            steps += callback.steps(kwargs.get("params", params), size)

        return steps

    def outputs(self, params: ImageParams, sources: int):
        outputs = sources
        for callback, _params, kwargs in self.stages:
            outputs = callback.outputs(kwargs.get("params", params), outputs)

        return outputs

    def __call__(
        self,
        worker: WorkerContext,
        server: ServerContext,
        params: ImageParams,
        sources: List[Image.Image],
        callback: Optional[ProgressCallback] = None,
        **pipeline_kwargs
    ) -> List[Image.Image]:
        """
        DEPRECATED: use `run` instead
        """
        if callback is None:
            callback = worker.get_progress_callback()
        else:
            callback = ChainProgress.from_progress(callback)

        start = monotonic()

        if len(sources) > 0:
            logger.info(
                "running pipeline on %s source images",
                len(sources),
            )
        else:
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

            per_stage_params = params
            if "params" in kwargs:
                per_stage_params = kwargs["params"]
                kwargs.pop("params")

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

            if stage_sources or must_tile:
                stage_outputs = []
                for source in stage_sources:
                    logger.info(
                        "image contains sources or is larger than tile size of %s, tiling stage",
                        tile,
                    )

                    extra_tiles = []

                    def stage_tile(
                        source_tile: Image.Image,
                        tile_mask: Image.Image,
                        dims: Tuple[int, int, int],
                    ) -> Image.Image:
                        for _i in range(worker.retries):
                            try:
                                output_tile = stage_pipe.run(
                                    worker,
                                    server,
                                    stage_params,
                                    per_stage_params,
                                    [source_tile],
                                    tile_mask=tile_mask,
                                    callback=callback,
                                    dims=dims,
                                    **kwargs,
                                )

                                if len(output_tile) > 1:
                                    while len(extra_tiles) < len(output_tile):
                                        extra_tiles.append([])

                                    for tile, layer in zip(output_tile, extra_tiles):
                                        layer.append((tile, dims))

                                if is_debug():
                                    save_image(server, "last-tile.png", output_tile[0])

                                return output_tile[0]
                            except Exception:
                                worker.retries = worker.retries - 1
                                logger.exception(
                                    "error while running stage pipeline for tile, %s retries left",
                                    worker.retries,
                                )
                                server.cache.clear()
                                run_gc([worker.get_device()])

                        raise RetryException("exhausted retries on tile")

                    output = process_tile_order(
                        stage_params.tile_order,
                        source,
                        tile,
                        stage_params.outscale,
                        [stage_tile],
                        **kwargs,
                    )

                    stage_outputs.append(output)

                    if len(extra_tiles) > 1:
                        for layer in extra_tiles:
                            layer_output = Image.new("RGB", output.size)
                            for layer_tile, dims in layer:
                                layer_output.paste(layer_tile, (dims[0], dims[1]))

                            stage_outputs.append(layer_output)

                stage_sources = stage_outputs
            else:
                logger.debug("image does not contain sources and is within tile size of %s, running stage", tile)
                for i in range(worker.retries):
                    try:
                        stage_outputs = stage_pipe.run(
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
                        stage_sources = stage_outputs
                        break
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
