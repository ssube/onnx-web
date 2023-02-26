from logging import getLogger
from torch.multiprocessing import Queue, Value
from typing import Any, Callable

from ..params import DeviceParams

logger = getLogger(__name__)


ProgressCallback = Callable[[int, int, Any], None]

class WorkerContext:
    cancel: "Value[bool]" = None
    key: str = None
    progress: "Value[int]" = None

    def __init__(
        self,
        key: str,
        cancel: "Value[bool]",
        device: DeviceParams,
        pending: "Queue[Any]",
        progress: "Value[int]",
    ):
        self.key = key
        self.cancel = cancel
        self.device = device
        self.pending = pending
        self.progress = progress

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
