from logging import getLogger
from os import getpid
from typing import Any, Callable, Optional

import numpy as np
from torch.multiprocessing import Queue, Value

from ..errors import CancelledException
from ..params import DeviceParams
from .command import JobCommand, JobStatus, ProgressCommand

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

    def get_progress(self) -> int:
        if self.last_progress is not None:
            return self.last_progress.steps

        return 0

    def get_progress_callback(self, reset=False) -> ProgressCallback:
        from ..chain.pipeline import ChainProgress

        if not reset and self.callback is not None:
            return self.callback

        def on_progress(step: int, timestep: int, latents: Any):
            self.callback.step = step
            self.set_progress(
                step,
                stages=self.callback.stage,
                tiles=self.callback.tile,
            )

        self.callback = ChainProgress.from_progress(on_progress)
        return self.callback

    def set_cancel(self, cancel: bool = True) -> None:
        with self.cancel.get_lock():
            self.cancel.value = cancel

    def set_idle(self, idle: bool = True) -> None:
        with self.idle.get_lock():
            self.idle.value = idle

    def set_progress(self, steps: int, stages: int = 0, tiles: int = 0) -> None:
        if self.job is None:
            raise RuntimeError("no job on which to set progress")

        if self.is_cancelled():
            raise CancelledException("job has been cancelled")

        result = None
        if self.callback is not None:
            result = self.callback.result

        logger.debug("setting progress for job %s to %s", self.job, steps)
        self.last_progress = ProgressCommand(
            self.job,
            self.job_type,
            self.device.device,
            JobStatus.RUNNING,
            steps=steps,
            stages=stages,
            tiles=tiles,
            result=result,
        )
        self.progress.put(
            self.last_progress,
            block=False,
        )

    def finish(self) -> None:
        if self.job is None:
            logger.warning("setting finished without an active job")
        else:
            logger.debug("setting finished for job %s", self.job)
            self.last_progress = ProgressCommand(
                self.job,
                self.job_type,
                self.device.device,
                JobStatus.SUCCESS,
                steps=self.callback.step,
                stages=self.callback.stage,
                tiles=self.callback.tile,
                result=self.callback.result,
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
                    steps=self.get_progress(),
                )
                self.progress.put(
                    self.last_progress,
                    block=False,
                )
            except Exception:
                logger.exception("error setting failure on job %s", self.job)
