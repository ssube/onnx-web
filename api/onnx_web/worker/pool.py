from collections import Counter
from logging import getLogger
from multiprocessing import Queue
from torch.multiprocessing import Lock, Process, Value
from typing import Callable, Dict, List, Optional, Tuple

from ..params import DeviceParams
from ..server import ServerContext
from .context import WorkerContext
from .worker import logger_init, worker_init

logger = getLogger(__name__)


class DevicePoolExecutor:
    devices: List[DeviceParams] = None
    finished: Dict[str, "Value[bool]"] = None
    pending: Dict[str, "Queue[WorkerContext]"] = None
    progress: Dict[str, "Value[int]"] = None
    workers: Dict[str, Process] = None
    jobs: Dict[str, str] = None

    def __init__(
        self,
        server: ServerContext,
        devices: List[DeviceParams],
        finished_limit: int = 10,
    ):
        self.server = server
        self.devices = devices
        self.finished = {}
        self.finished_limit = finished_limit
        self.context = {}
        self.locks = {}
        self.pending = {}
        self.progress = {}
        self.workers = {}
        self.jobs = {} # Dict[Output, Device]
        self.job_count = 0

        # TODO: make this a method
        logger.debug("starting log worker")
        self.log_queue = Queue()
        log_lock = Lock()
        self.locks["logger"] = log_lock
        self.logger = Process(target=logger_init, args=(log_lock, self.log_queue))
        self.logger.start()

        logger.debug("testing log worker")
        self.log_queue.put("testing")

        # create a pending queue and progress value for each device
        for device in devices:
            name = device.device
            # TODO: make this a method
            lock = Lock()
            self.locks[name] = lock
            cancel = Value("B", False, lock=lock)
            finished = Value("B", False)
            self.finished[name] = finished
            progress = Value("I", 0) # , lock=lock) # needs its own lock for some reason. TODO: why?
            self.progress[name] = progress
            pending = Queue()
            self.pending[name] = pending
            context = WorkerContext(name, cancel, device, pending, progress, self.log_queue, finished)
            self.context[name] = context

            logger.debug("starting worker for device %s", device)
            self.workers[name] = Process(target=worker_init, args=(lock, context, server))
            self.workers[name].start()

    def cancel(self, key: str) -> bool:
        """
        Cancel a job. If the job has not been started, this will cancel
        the future and never execute it. If the job has been started, it
        should be cancelled on the next progress callback.
        """
        raise NotImplementedError()

    def done(self, key: str) -> Tuple[Optional[bool], int]:
        if not key in self.jobs:
            logger.warn("checking status for unknown key: %s", key)
            return (None, 0)

        device = self.jobs[key]
        finished = self.finished[device]
        progress = self.progress[device]

        return (finished.value, progress.value)


    def get_next_device(self, needs_device: Optional[DeviceParams] = None) -> int:
        # respect overrides if possible
        if needs_device is not None:
            for i in range(len(self.devices)):
                if self.devices[i].device == needs_device.device:
                    return i

        pending = [
            self.pending[d.device].qsize() for d in self.devices
        ]
        jobs = Counter(range(len(self.devices)))
        jobs.update(pending)

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
                worker.join(5)

        if self.logger.is_alive():
            self.logger.join(5)

    def prune(self):
        finished_count = len(self.finished)
        if finished_count > self.finished_limit:
            logger.debug(
                "pruning %s of %s finished jobs",
                finished_count - self.finished_limit,
                finished_count,
            )
            self.finished[:] = self.finished[-self.finished_limit:]

    def recycle(self):
        for name, proc in self.workers.items():
            if proc.is_alive():
                logger.debug("shutting down worker for device %s", name)
                proc.join(5)
            else:
                logger.warning("worker for device %s has died", name)

            self.workers[name] = None

        logger.info("starting new workers")

        for name in self.workers.keys():
            context = self.context[name]
            lock = self.locks[name]

            logger.debug("starting worker for device %s", name)
            self.workers[name] = Process(target=worker_init, args=(lock, context, self.server))
            self.workers[name].start()


    def submit(
        self,
        key: str,
        fn: Callable[..., None],
        /,
        *args,
        needs_device: Optional[DeviceParams] = None,
        **kwargs,
    ) -> None:
        self.job_count += 1
        if self.job_count > 10:
            self.recycle()
            self.job_count = 0

        self.prune()
        device_idx = self.get_next_device(needs_device=needs_device)
        logger.info(
            "assigning job %s to device %s: %s", key, device_idx, self.devices[device_idx]
        )

        device = self.devices[device_idx]
        queue = self.pending[device.device]
        queue.put((fn, args, kwargs))

        self.jobs[key] = device.device


    def status(self) -> List[Tuple[str, int, bool, int]]:
        pending = [
            (
                device.device,
                self.pending[device.device].qsize(),
                self.progress[device.device].value,
                self.workers[device.device].is_alive(),
            )
            for device in self.devices
        ]
        pending.extend(self.finished)
        return pending
