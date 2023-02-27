from logging import getLogger
from typing import Any, Callable, Tuple

from torch.multiprocessing import Queue, Value

from ..params import DeviceParams

logger = getLogger(__name__)


ProgressCallback = Callable[[int, int, Any], None]


class WorkerContext:
    cancel: "Value[bool]" = None
    key: str = None
    pending: "Queue[Tuple[Callable, Any, Any]]" = None
    progress: "Value[int]" = None

    def __init__(
        self,
        key: str,
        device: DeviceParams,
        cancel: "Value[bool]" = None,
        progress: "Value[int]" = None,
        finished: "Queue[str]" = None,
        logs: "Queue[str]" = None,
        pending: "Queue[Any]" = None,
        started: "Queue[Tuple[str, str]]" = None,
    ):
        self.key = key
        self.device = device
        self.cancel = cancel
        self.progress = progress
        self.finished = finished
        self.logs = logs
        self.pending = pending
        self.started = started

    def is_cancelled(self) -> bool:
        return self.cancel.value

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
            if self.is_cancelled():
                raise RuntimeError("job has been cancelled")
            else:
                logger.debug("setting progress for job %s to %s", self.key, step)
                self.set_progress(step)

        return on_progress

    def set_cancel(self, cancel: bool = True) -> None:
        with self.cancel.get_lock():
            self.cancel.value = cancel

    def set_progress(self, progress: int) -> None:
        with self.progress.get_lock():
            self.progress.value = progress

    def put_finished(self, job: str) -> None:
        self.finished.put((job, self.device.device))

    def put_started(self, job: str) -> None:
        self.started.put((job, self.device.device))

    def clear_flags(self) -> None:
        self.set_cancel(False)
        self.set_progress(0)
