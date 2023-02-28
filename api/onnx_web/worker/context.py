from logging import getLogger
from typing import Any, Callable, Tuple

from torch.multiprocessing import Queue, Value

from ..params import DeviceParams

logger = getLogger(__name__)


ProgressCallback = Callable[[int, int, Any], None]


class WorkerContext:
    cancel: "Value[bool]" = None
    job: str = None
    pending: "Queue[Tuple[Callable, Any, Any]]" = None
    progress: "Value[int]" = None

    def __init__(
        self,
        job: str,
        device: DeviceParams,
        cancel: "Value[bool]" = None,
        logs: "Queue[str]" = None,
        pending: "Queue[Any]" = None,
        progress: "Queue[Tuple[str, int]]" = None,
        finished: "Queue[str]" = None,
    ):
        self.job = job
        self.device = device
        self.cancel = cancel
        self.progress = progress
        self.finished = finished
        self.logs = logs
        self.pending = pending

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
                logger.debug("setting progress for job %s to %s", self.job, step)
                self.set_progress(step)

        return on_progress

    def set_cancel(self, cancel: bool = True) -> None:
        with self.cancel.get_lock():
            self.cancel.value = cancel

    def set_progress(self, progress: int) -> None:
        self.progress.put((self.job, self.device.device, progress), block=False)

    def set_finished(self) -> None:
        self.finished.put((self.job, self.device.device))

    def clear_flags(self) -> None:
        self.set_cancel(False)
        self.set_progress(0)


class JobStatus:
    def __init__(
        self,
        name: str,
        progress: int = 0,
        cancelled: bool = False,
        finished: bool = False,
    ) -> None:
        self.name = name
        self.progress = progress
        self.cancelled = cancelled
        self.finished = finished
