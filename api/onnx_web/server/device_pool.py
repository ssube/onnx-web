from collections import Counter
from concurrent.futures import Future
from logging import getLogger
from multiprocessing import Queue
from torch.multiprocessing import Lock, Process, SimpleQueue, Value
from traceback import format_exception
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from time import sleep

from ..params import DeviceParams
from ..utils import run_gc

logger = getLogger(__name__)

ProgressCallback = Callable[[int, int, Any], None]


def worker_init(lock: Lock, job_queue: SimpleQueue):
    logger.info("checking in from worker")

    while True:
        if job_queue.empty():
            logger.info("no jobs, sleeping")
            sleep(5)
        else:
            job = job_queue.get()
            logger.info("got job: %s", job)


class JobContext:
    cancel: Value = None
    device_index: Value = None
    devices: List[DeviceParams] = None
    key: str = None
    progress: Value = None

    def __init__(
        self,
        key: str,
        devices: List[DeviceParams],
        cancel: bool = False,
        device_index: int = -1,
        progress: int = 0,
    ):
        self.key = key
        self.devices = list(devices)
        self.cancel = Value("B", cancel)
        self.device_index = Value("i", device_index)
        self.progress = Value("I", progress)

    def is_cancelled(self) -> bool:
        return self.cancel.value

    def get_device(self) -> DeviceParams:
        """
        Get the device assigned to this job.
        """
        with self.device_index.get_lock():
            device_index = self.device_index.value
            if device_index < 0:
                raise ValueError("job has not been assigned to a device")
            else:
                device = self.devices[device_index]
                logger.debug("job %s assigned to device %s", self.key, device)
                return device

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


class Job:
    """
    Link a future to its context.
    """

    context: JobContext = None
    future: Future = None
    key: str = None

    def __init__(
        self,
        key: str,
        future: Future,
        context: JobContext,
    ):
        self.context = context
        self.future = future
        self.key = key

    def get_progress(self) -> int:
        return self.context.get_progress()

    def set_cancel(self, cancel: bool = True):
        return self.context.set_cancel(cancel)

    def set_progress(self, progress: int):
        return self.context.set_progress(progress)


class DevicePoolExecutor:
    devices: List[DeviceParams] = None
    finished: List[Tuple[str, int]] = None
    pending: Dict[str, "Queue[Job]"] = None
    progress: Dict[str, Value] = None
    workers: Dict[str, Process] = None

    def __init__(
        self,
        devices: List[DeviceParams],
        finished_limit: int = 10,
    ):
        self.devices = devices
        self.finished = []
        self.finished_limit = finished_limit
        self.lock = Lock()
        self.pending = {}
        self.progress = {}
        self.workers = {}

        # create a pending queue and progress value for each device
        for device in devices:
            name = device.device
            job_queue = Queue()
            self.pending[name] = job_queue
            self.progress[name] = Value("I", 0, lock=self.lock)
            self.workers[name] = Process(target=worker_init, args=(self.lock, job_queue))

    def cancel(self, key: str) -> bool:
        """
        Cancel a job. If the job has not been started, this will cancel
        the future and never execute it. If the job has been started, it
        should be cancelled on the next progress callback.
        """
        raise NotImplementedError()

    def done(self, key: str) -> Tuple[Optional[bool], int]:
        for k, progress in self.finished:
            if key == k:
                return (True, progress)

        logger.warn("checking status for unknown key: %s", key)
        return (None, 0)

    def get_next_device(self, needs_device: Optional[DeviceParams] = None) -> int:
        # respect overrides if possible
        if needs_device is not None:
            for i in range(len(self.devices)):
                if self.devices[i].device == needs_device.device:
                    return i

        # use the first/default device if there are no jobs
        if len(self.jobs) == 0:
            return 0

        job_devices = [
            job.context.device_index.value for job in self.jobs if not job.future.done()
        ]
        job_counts = Counter(range(len(self.devices)))
        job_counts.update(job_devices)

        queued = job_counts.most_common()
        logger.debug("jobs queued by device: %s", queued)

        lowest_count = queued[-1][1]
        lowest_devices = [d[0] for d in queued if d[1] == lowest_count]
        lowest_devices.sort()

        return lowest_devices[0]

    def prune(self):
        finished_count = len(self.finished)
        if finished_count > self.finished_limit:
            logger.debug(
                "pruning %s of %s finished jobs",
                finished_count - self.finished_limit,
                finished_count,
            )
            self.finished[:] = self.finished[-self.finished_limit:]

    def submit(
        self,
        key: str,
        fn: Callable[..., None],
        /,
        *args,
        needs_device: Optional[DeviceParams] = None,
        **kwargs,
    ) -> None:
        self.prune()
        device_idx = self.get_next_device(needs_device=needs_device)
        logger.info(
            "assigning job %s to device %s: %s", key, device_idx, self.devices[device_idx]
        )

        context = JobContext(key, self.devices, device_index=device_idx)
        device = self.devices[device_idx]

        queue = self.pending[device.device]
        queue.put((fn, context, args, kwargs))


    def status(self) -> List[Tuple[str, int, bool, int]]:
        pending = [
            (
                device.device,
                self.pending[device.device].qsize(),
            )
            for device in self.devices
        ]
        pending.extend(self.finished)
        return pending
