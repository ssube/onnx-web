from logging import getLogger
from os import getpid
from typing import Any, Callable, Optional

import numpy as np
from torch.multiprocessing import Queue, Value

from ..errors import CancelledException
from ..params import DeviceParams
from .command import JobCommand, JobStatus, Progress, ProgressCommand

logger = getLogger(__name__)


ProgressCallback = Callable[[int, int, np.ndarray], None]


class WorkerContext:
    cancel: "Value[bool]"
    job: Optional[str]
    job_type: Optional[str]
    name: str
    pending: "Queue[JobCommand]"
    active_pid: "Value[int]"
    progress: "Queue[ProgressCommand]"
    last_progress: Optional[ProgressCommand]
    idle: "Value[bool]"
    timeout: float
    retries: int
    initial_retries: int
    callback: Optional[Any]

    # progress state
    steps: Progress
    stages: Progress
    tiles: Progress

    def __init__(
        self,
        name: str,
        device: DeviceParams,
        cancel: "Value[bool]",
        logs: "Queue[str]",
        pending: "Queue[JobCommand]",
        progress: "Queue[ProgressCommand]",
        active_pid: "Value[int]",
        idle: "Value[bool]",
        retries: int,
        timeout: float,
    ):
        self.job = None
        self.job_type = None
        self.name = name
        self.device = device
        self.cancel = cancel
        self.progress = progress
        self.logs = logs
        self.pending = pending
        self.active_pid = active_pid
        self.last_progress = None
        self.idle = idle
        self.initial_retries = retries
        self.retries = retries
        self.timeout = timeout
        self.callback = None
        self.steps = Progress(0, 0)
        self.stages = Progress(0, 0)
        self.tiles = Progress(0, 0)

    def start(self, job: JobCommand) -> None:
        # set job name and type
        self.job = job.name
        self.job_type = job.job_type

        # reset retries
        self.retries = self.initial_retries

        # clear flags
        self.set_cancel(cancel=False)
        self.set_idle(idle=False)

    def is_active(self) -> bool:
        return self.get_active() == getpid()

    def is_cancelled(self) -> bool:
        return self.cancel.value

    def is_idle(self) -> bool:
        return self.idle.value

    def get_active(self) -> int:
        with self.active_pid.get_lock():
            return self.active_pid.value

    def get_device(self) -> DeviceParams:
        """
        Get the device assigned to this job.
        """
        return self.device

    def get_progress(self) -> Progress:
        return self.get_last_steps()

    def get_last_steps(self) -> Progress:
        return self.steps

    def get_last_stages(self) -> Progress:
        return self.stages

    def get_last_tiles(self) -> Progress:
        return self.tiles

    def get_progress_callback(self, reset=False) -> ProgressCallback:
        from ..chain.pipeline import ChainProgress

        if not reset and self.callback is not None:
            return self.callback

        def on_progress(step: int, timestep: int, latents: Any):
            self.set_progress(
                step,
            )

        self.callback = ChainProgress.from_progress(on_progress)
        return self.callback

    def set_cancel(self, cancel: bool = True) -> None:
        with self.cancel.get_lock():
            self.cancel.value = cancel

    def set_idle(self, idle: bool = True) -> None:
        with self.idle.get_lock():
            self.idle.value = idle

    def set_progress(self, steps: int, stages: int = None, tiles: int = None) -> None:
        if self.job is None:
            raise RuntimeError("no job on which to set progress")

        if self.is_cancelled():
            raise CancelledException("job has been cancelled")

        # update current progress counters
        self.steps.current = steps

        if stages is not None:
            self.stages.current = stages

        if tiles is not None:
            self.tiles.current = tiles

        # TODO: result should really be part of context at this point
        result = None
        if self.callback is not None:
            result = self.callback.result

        # send progress to worker pool
        logger.debug("setting progress for job %s to %s", self.job, steps)
        self.last_progress = ProgressCommand(
            self.job,
            self.job_type,
            self.device.device,
            JobStatus.RUNNING,
            steps=self.steps,
            stages=self.stages,
            tiles=self.tiles,
            result=result,
        )
        self.progress.put(
            self.last_progress,
            block=False,
        )

    def set_steps(self, current: int, total: int = 0) -> None:
        if total > 0:
            self.steps = Progress(current, total)
        else:
            self.steps.current = current

    def set_stages(self, current: int, total: int = 0) -> None:
        if total > 0:
            self.stages = Progress(current, total)
        else:
            self.stages.current = current

    def set_tiles(self, current: int, total: int = 0) -> None:
        if total > 0:
            self.tiles = Progress(current, total)
        else:
            self.tiles.current = current

    def set_totals(self, steps: int, stages: int = 0, tiles: int = 0) -> None:
        self.steps.total = steps
        self.stages.total = stages
        self.tiles.total = tiles

    def finish(self) -> None:
        if self.job is None:
            logger.warning("setting finished without an active job")
        else:
            logger.debug("setting finished for job %s", self.job)

            result = None
            if self.callback is not None:
                result = self.callback.result

            self.last_progress = ProgressCommand(
                self.job,
                self.job_type,
                self.device.device,
                JobStatus.SUCCESS,
                steps=self.steps,
                stages=self.stages,
                tiles=self.tiles,
                result=result,
            )
            self.progress.put(
                self.last_progress,
                block=False,
            )

    def fail(self) -> None:
        if self.job is None:
            logger.warning("setting failure without an active job")
        else:
            logger.warning("setting failure for job %s", self.job)
            try:
                self.last_progress = ProgressCommand(
                    self.job,
                    self.job_type,
                    self.device.device,
                    JobStatus.FAILED,
                    steps=self.steps,
                    stages=self.stages,
                    tiles=self.tiles,
                    # TODO: should this include partial results?
                )
                self.progress.put(
                    self.last_progress,
                    block=False,
                )
            except Exception:
                logger.exception("error setting failure on job %s", self.job)
