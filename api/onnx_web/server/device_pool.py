from collections import Counter
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from logging import getLogger
from multiprocessing import Value
from traceback import format_exception
from typing import Any, Callable, List, Optional, Tuple, Union

from ..params import DeviceParams
from ..utils import run_gc

logger = getLogger(__name__)

ProgressCallback = Callable[[int, int, Any], None]


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
                raise Exception("job has not been assigned to a device")
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
                raise Exception("job has been cancelled")
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
    jobs: List[Job] = None
    next_device: int = 0
    pool: Union[ProcessPoolExecutor, ThreadPoolExecutor] = None
    recent: List[Tuple[str, int]] = None

    def __init__(
        self,
        devices: List[DeviceParams],
        pool: Optional[Union[ProcessPoolExecutor, ThreadPoolExecutor]] = None,
        recent_limit: int = 10,
    ):
        self.devices = devices
        self.jobs = []
        self.next_device = 0
        self.recent = []
        self.recent_limit = recent_limit

        device_count = len(devices)
        if pool is None:
            logger.info(
                "creating thread pool executor for %s devices: %s",
                device_count,
                [d.device for d in devices],
            )
            self.pool = ThreadPoolExecutor(device_count)
        else:
            logger.info(
                "using existing pool for %s devices: %s",
                device_count,
                [d.device for d in devices],
            )
            self.pool = pool

    def cancel(self, key: str) -> bool:
        """
        Cancel a job. If the job has not been started, this will cancel
        the future and never execute it. If the job has been started, it
        should be cancelled on the next progress callback.
        """
        for job in self.jobs:
            if job.key == key:
                if job.future.cancel():
                    return True
                else:
                    job.set_cancel()
                    return True

        return False

    def done(self, key: str) -> Tuple[Optional[bool], int]:
        for k, progress in self.recent:
            if key == k:
                return (True, progress)

        for job in self.jobs:
            if job.key == key:
                done = job.future.done()
                progress = job.get_progress()

                if done:
                    self.prune()

                return (done, progress)

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
        pending_jobs = [job for job in self.jobs if job.future.done()]
        logger.debug("pruning %s of %s pending jobs", len(pending_jobs), len(self.jobs))

        for job in pending_jobs:
            self.recent.append((job.key, job.get_progress()))
            try:
                self.jobs.remove(job)
            except ValueError as e:
                logger.warning("error removing pruned job from pending: %s", e)

        recent_count = len(self.recent)
        if recent_count > self.recent_limit:
            logger.debug(
                "pruning %s of %s recent jobs",
                recent_count - self.recent_limit,
                recent_count,
            )
            self.recent[:] = self.recent[-self.recent_limit :]

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
        device = self.get_next_device(needs_device=needs_device)
        logger.info(
            "assigning job %s to device %s: %s", key, device, self.devices[device]
        )

        context = JobContext(key, self.devices, device_index=device)
        future = self.pool.submit(fn, context, *args, **kwargs)
        job = Job(key, future, context)
        self.jobs.append(job)

        def job_done(f: Future):
            try:
                f.result()
                logger.info("job %s finished successfully", key)
            except Exception as err:
                logger.warn(
                    "job %s failed with an error: %s",
                    key,
                    format_exception(type(err), err, err.__traceback__),
                )
                run_gc([self.devices[device]])

        future.add_done_callback(job_done)

    def status(self) -> List[Tuple[str, int, bool, int]]:
        pending = [
            (
                job.key,
                job.context.device_index.value,
                job.future.done(),
                job.get_progress(),
            )
            for job in self.jobs
        ]
        recent = [
            (
                key,
                None,
                True,
                progress,
            )
            for key, progress in self.recent
        ]

        pending.extend(recent)
        return pending
