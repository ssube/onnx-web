from logging import getLogger
from os import getpid
from typing import Any, Callable, Optional

from torch.multiprocessing import Queue, Value

from ..params import DeviceParams
from .command import JobCommand, ProgressCommand

logger = getLogger(__name__)


ProgressCallback = Callable[[int, int, Any], None]


class WorkerContext:
    cancel: "Value[bool]"
    job: str
    pending: "Queue[JobCommand]"
    active_pid: "Value[int]"
    progress: "Queue[ProgressCommand]"
    last_progress: Optional[ProgressCommand]
    idle: "Value[bool]"
    timeout: float
    retries: int

    def __init__(
        self,
        job: str,
        device: DeviceParams,
        cancel: "Value[bool]",
        logs: "Queue[str]",
        pending: "Queue[JobCommand]",
        progress: "Queue[ProgressCommand]",
        active_pid: "Value[int]",
        idle: "Value[bool]",
    ):
        self.job = job
        self.device = device
        self.cancel = cancel
        self.progress = progress
        self.logs = logs
        self.pending = pending
        self.active_pid = active_pid
        self.last_progress = None
        self.idle = idle
        self.timeout = 1.0
        self.retries = 3  # TODO: get from env

    def start(self, job: str) -> None:
        self.job = job
        self.retries = 3
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
            return self.last_progress.progress

        return 0

    def get_progress_callback(self) -> ProgressCallback:
        from ..chain.base import ChainProgress

        def on_progress(step: int, timestep: int, latents: Any):
            on_progress.step = step
            self.set_progress(step)

        return ChainProgress.from_progress(on_progress)

    def set_cancel(self, cancel: bool = True) -> None:
        with self.cancel.get_lock():
            self.cancel.value = cancel

    def set_idle(self, idle: bool = True) -> None:
        with self.idle.get_lock():
            self.idle.value = idle

    def set_progress(self, progress: int) -> None:
        if self.is_cancelled():
            raise RuntimeError("job has been cancelled")
        else:
            logger.debug("setting progress for job %s to %s", self.job, progress)
            self.last_progress = ProgressCommand(
                self.job,
                self.device.device,
                False,
                progress,
                self.is_cancelled(),
                False,
            )

            self.progress.put(
                self.last_progress,
                block=False,
            )

    def finish(self) -> None:
        logger.debug("setting finished for job %s", self.job)
        self.last_progress = ProgressCommand(
            self.job,
            self.device.device,
            True,
            self.get_progress(),
            self.is_cancelled(),
            False,
        )
        self.progress.put(
            self.last_progress,
            block=False,
        )

    def fail(self) -> None:
        logger.warning("setting failure for job %s", self.job)
        try:
            self.last_progress = ProgressCommand(
                self.job,
                self.device.device,
                True,
                self.get_progress(),
                self.is_cancelled(),
                True,
            )
            self.progress.put(
                self.last_progress,
                block=False,
            )
        except Exception:
            logger.exception("error setting failure on job %s", self.job)


class JobStatus:
    name: str
    device: str
    progress: int
    cancelled: bool
    finished: bool

    def __init__(
        self,
        name: str,
        device: DeviceParams,
        progress: int = 0,
        cancelled: bool = False,
        finished: bool = False,
    ) -> None:
        self.name = name
        self.device = device.device
        self.progress = progress
        self.cancelled = cancelled
        self.finished = finished
