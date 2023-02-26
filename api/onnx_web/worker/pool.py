from collections import Counter
from logging import getLogger
from multiprocessing import Queue
from typing import Callable, Dict, List, Optional, Tuple

from torch.multiprocessing import Process, Value

from ..params import DeviceParams
from ..server import ServerContext
from .context import WorkerContext
from .worker import logger_init, worker_init

logger = getLogger(__name__)


class DevicePoolExecutor:
    context: Dict[str, WorkerContext] = None
    devices: List[DeviceParams] = None
    pending: Dict[str, "Queue[WorkerContext]"] = None
    workers: Dict[str, Process] = None
    active_job: Dict[str, str] = None
    finished: List[Tuple[str, int, bool]] = None

    def __init__(
        self,
        server: ServerContext,
        devices: List[DeviceParams],
        max_jobs_per_worker: int = 10,
        join_timeout: float = 5.0,
    ):
        self.server = server
        self.devices = devices
        self.max_jobs_per_worker = max_jobs_per_worker
        self.join_timeout = join_timeout

        self.context = {}
        self.pending = {}
        self.workers = {}
        self.active_job = {}
        self.finished_jobs = 0  # TODO: turn this into a Dict per-worker

        self.create_logger_worker()
        for device in devices:
            self.create_device_worker(device)

        logger.debug("testing log worker")
        self.log_queue.put("testing")

    def create_logger_worker(self) -> None:
        self.log_queue = Queue()
        self.logger = Process(target=logger_init, args=(self.log_queue))

        logger.debug("starting log worker")
        self.logger.start()

    def create_device_worker(self, device: DeviceParams) -> None:
        name = device.device
        pending = Queue()
        self.pending[name] = pending
        context = WorkerContext(
            name,
            device,
            cancel=Value("B", False),
            finished=Value("B", False),
            progress=Value("I", 0),
            pending=pending,
            logs=self.log_queue,
        )
        self.context[name] = context
        self.workers[name] = Process(target=worker_init, args=(context, self.server))

        logger.debug("starting worker for device %s", device)
        self.workers[name].start()

    def create_prune_worker(self) -> None:
        # TODO: create a background thread to prune completed jobs
        pass

    def cancel(self, key: str) -> bool:
        """
        Cancel a job. If the job has not been started, this will cancel
        the future and never execute it. If the job has been started, it
        should be cancelled on the next progress callback.
        """
        if key not in self.active_job:
            logger.warn("attempting to cancel unknown job: %s", key)
            return False

        device = self.active_job[key]
        context = self.context[device]
        logger.info("cancelling job %s on device %s", key, device)

        if context.cancel.get_lock():
            context.cancel.value = True

        # self.finished.append((key, context.progress.value, context.cancel.value)) maybe?
        return True

    def done(self, key: str) -> Tuple[Optional[bool], int]:
        if key not in self.active_job:
            logger.warn("checking status for unknown job: %s", key)
            return (None, 0)

        # TODO: prune here, maybe?

        device = self.active_job[key]
        context = self.context[device]

        if context.finished.value is True:
            self.finished.append((key, context.progress.value, context.cancel.value))

        return (context.finished.value, context.progress.value)

    def get_next_device(self, needs_device: Optional[DeviceParams] = None) -> int:
        # respect overrides if possible
        if needs_device is not None:
            for i in range(len(self.devices)):
                if self.devices[i].device == needs_device.device:
                    return i

        jobs = Counter(range(len(self.devices)))
        jobs.update([self.pending[d.device].qsize() for d in self.devices])

        queued = jobs.most_common()
        logger.debug("jobs queued by device: %s", queued)

        lowest_count = queued[-1][1]
        lowest_devices = [d[0] for d in queued if d[1] == lowest_count]
        lowest_devices.sort()

        return lowest_devices[0]

    def join(self):
        for device, worker in self.workers.items():
            if worker.is_alive():
                logger.info("stopping worker for device %s", device)
                worker.join(self.join_timeout)

        if self.logger.is_alive():
            self.logger.join(self.join_timeout)

    def recycle(self):
        for name, proc in self.workers.items():
            if proc.is_alive():
                logger.debug("shutting down worker for device %s", name)
                proc.join(self.join_timeout)
                proc.terminate()
            else:
                logger.warning("worker for device %s has died", name)

            self.workers[name] = None
            del proc

        logger.info("starting new workers")

        for device in self.devices:
            self.create_device_worker(device)

    def submit(
        self,
        key: str,
        fn: Callable[..., None],
        /,
        *args,
        needs_device: Optional[DeviceParams] = None,
        **kwargs,
    ) -> None:
        self.finished_jobs += 1
        logger.debug("pool job count: %s", self.finished_jobs)
        if self.finished_jobs > self.max_jobs_per_worker:
            self.recycle()
            self.finished_jobs = 0

        device_idx = self.get_next_device(needs_device=needs_device)
        logger.info(
            "assigning job %s to device %s: %s",
            key,
            device_idx,
            self.devices[device_idx],
        )

        device = self.devices[device_idx]
        queue = self.pending[device.device]
        queue.put((fn, args, kwargs))

        self.active_job[key] = device.device

    def status(self) -> List[Tuple[str, int, bool, int]]:
        pending = [
            (
                name,
                self.workers[name].is_alive(),
                context.pending.qsize(),
                context.cancel.value,
                context.finished.value,
                context.progress.value,
            )
            for name, context in self.context.items()
        ]
        pending.extend(self.finished)
        return pending
