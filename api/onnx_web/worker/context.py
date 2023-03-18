from logging import getLogger
from os import getpid
from typing import Any, Callable

from torch.multiprocessing import Queue, Value

from ..params import DeviceParams
from .command import JobCommand, ProgressCommand

logger = getLogger(__name__)


ProgressCallback = Callable[[int, int, Any], None]


class WorkerContext:
    cancel: "Value[bool]"
    job: str
    pending: "Queue[JobCommand]"
    current: "Value[int]"
    progress: "Queue[ProgressCommand]"

    def __init__(
        self,
        job: str,
        device: DeviceParams,
        cancel: "Value[bool]",
        logs: "Queue[str]",
        pending: "Queue[JobCommand]",
        progress: "Queue[ProgressCommand]",
        current: "Value[int]",
    ):
        self.job = job
        self.device = device
        self.cancel = cancel
        self.progress = progress
        self.logs = logs
        self.pending = pending
        self.current = current

    def is_cancelled(self) -> bool:
        return self.cancel.value

    def is_current(self) -> bool:
        return self.get_current() == getpid()

    def get_current(self) -> int:
        with self.current.get_lock():
            return self.current.value

    def get_device(self) -> DeviceParams:
        """
        Get the device assigned to this job.
        """
        return self.device

    def get_progress(self) -> int:
        return self.progress.value

    def get_progress_callback(self) -> ProgressCallback:
        def on_progress(step: int, timestep: int, latents: Any):
            on_progress.step = step
            self.set_progress(step)

        return on_progress

    def set_cancel(self, cancel: bool = True) -> None:
        with self.cancel.get_lock():
            self.cancel.value = cancel

    def set_progress(self, progress: int) -> None:
        if self.is_cancelled():
            raise RuntimeError("job has been cancelled")
        else:
            logger.debug("setting progress for job %s to %s", self.job, progress)
            self.progress.put(
                ProgressCommand(
                    self.job,
                    self.device.device,
                    False,
                    progress,
                    self.is_cancelled(),
                    False,
                ),
                block=False,
            )

    def set_finished(self) -> None:
        logger.debug("setting finished for job %s", self.job)
        self.progress.put(
            ProgressCommand(
                self.job,
                self.device.device,
                True,
                self.get_progress(),
                self.is_cancelled(),
                False,
            ),
            block=False,
        )

    def set_failed(self) -> None:
        logger.warning("setting failure for job %s", self.job)
        try:
            self.progress.put(
                ProgressCommand(
                    self.job,
                    self.device.device,
                    True,
                    self.get_progress(),
                    self.is_cancelled(),
                    True,
                ),
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
