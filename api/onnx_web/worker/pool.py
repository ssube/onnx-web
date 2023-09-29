from collections import Counter
from logging import getLogger
from queue import Empty
from threading import Lock, Thread
from typing import Callable, Dict, List, Optional, Tuple

from torch.multiprocessing import Process, Queue, Value

from ..params import DeviceParams
from ..server import ServerContext
from .command import JobCommand, ProgressCommand
from .context import WorkerContext
from .utils import Interval
from .worker import worker_main

logger = getLogger(__name__)


class DevicePoolExecutor:
    server: ServerContext
    devices: List[DeviceParams]

    join_timeout: float
    max_jobs_per_worker: int
    max_pending_per_worker: int
    progress_interval: float
    recycle_interval: float

    leaking: List[Tuple[str, Process, WorkerContext]]

    worker_cancel: Dict[str, "Value[bool]"]
    worker_idle: Dict[str, "Value[bool]"]

    context: Dict[str, WorkerContext]  # Device -> Context
    current: Dict[str, "Value[int]"]  # Device -> pid
    pending: Dict[str, "Queue[JobCommand]"]
    progress: Dict[str, "Queue[ProgressCommand]"]
    workers: Dict[str, Process]

    health_worker: Interval
    logger_worker: Thread
    progress_worker: Interval

    cancelled_jobs: List[str]
    finished_jobs: List[ProgressCommand]
    pending_jobs: List[JobCommand]
    running_jobs: Dict[str, ProgressCommand]  # Device -> job progress
    total_jobs: Dict[str, int]  # Device -> job count

    logs: "Queue[str]"
    rlock: Lock

    def __init__(
        self,
        server: ServerContext,
        devices: List[DeviceParams],
        max_pending_per_worker: int = 100,
        join_timeout: float = 5.0,
        recycle_interval: float = 10,
        progress_interval: float = 1.0,
    ):
        self.server = server
        self.devices = devices

        self.join_timeout = join_timeout
        self.max_jobs_per_worker = server.job_limit
        self.max_pending_per_worker = max_pending_per_worker
        self.progress_interval = progress_interval
        self.recycle_interval = recycle_interval

        self.leaking = []
        self.context = {}
        self.current = {}
        self.pending = {}
        self.progress = {}
        self.workers = {}

        self.cancelled_jobs = []
        self.finished_jobs = []
        self.pending_jobs = []
        self.running_jobs = {}
        self.total_jobs = {}
        self.worker_cancel = {}
        self.worker_idle = {}

        self.logs = Queue(self.max_pending_per_worker)
        self.rlock = Lock()

    def start(self, *args) -> None:
        self.create_health_worker()
        self.create_logger_worker()
        self.create_progress_worker()

        for device in self.devices:
            self.create_device_worker(device, *args)

    def create_device_worker(self, device: DeviceParams, *args) -> None:
        name = device.device

        # always recreate queues
        self.progress[name] = Queue(self.max_pending_per_worker)
        self.pending[name] = Queue(self.max_pending_per_worker)
        self.total_jobs[device.device] = 0

        # reuse pid sentinel
        if name in self.current:
            logger.debug("using existing current worker value")
            current = self.current[name]
        else:
            logger.debug("creating new current worker value")
            current = Value("L", 0)
            self.current[name] = current

        self.worker_cancel[name] = Value("B", False)
        self.worker_idle[name] = Value("B", False)

        # create a new context and worker
        context = WorkerContext(
            name,
            device,
            cancel=self.worker_cancel[name],
            progress=self.progress[name],
            logs=self.logs,
            pending=self.pending[name],
            active_pid=current,
            idle=self.worker_idle[name],
            retries=self.server.worker_retries,
            timeout=self.progress_interval,
        )
        self.context[name] = context

        worker = Process(
            name=f"onnx-web worker: {name}",
            target=worker_main,
            args=(context, self.server, *args),
            daemon=True,
        )
        self.workers[name] = worker

        logger.debug("starting worker for device %s", device)
        worker.start()
        current.value = worker.pid

    def create_health_worker(self) -> None:
        self.health_worker = Interval(self.recycle_interval, health_main, args=(self,))
        self.health_worker.daemon = True
        self.health_worker.name = "onnx-web health"
        logger.debug("starting health worker")
        self.health_worker.start()

    def create_logger_worker(self) -> None:
        self.logger_worker = Thread(
            name="onnx-web logger",
            target=logger_main,
            args=(
                self,
                self.logs,
            ),
            daemon=True,
        )

        logger.debug("starting logger worker")
        self.logger_worker.start()

    def create_progress_worker(self) -> None:
        self.progress_worker = Interval(
            self.progress_interval, progress_main, args=(self,)
        )
        self.progress_worker.daemon = True
        self.progress_worker.name = "onnx-web progress"
        logger.debug("starting progress worker")
        self.progress_worker.start()

    def get_job_context(self, key: str) -> WorkerContext:
        device, _progress = self.running_jobs[key]
        return self.context[device]

    def get_next_device(self, needs_device: Optional[DeviceParams] = None) -> int:
        # respect overrides if possible
        if needs_device is not None:
            for i in range(len(self.devices)):
                if self.devices[i].device == needs_device.device:
                    return i

        jobs = Counter(range(len(self.devices)))
        jobs.update([self.pending[d.device].qsize() for d in self.devices])

        queued = jobs.most_common()
        logger.trace("jobs queued by device: %s", queued)

        lowest_count = queued[-1][1]
        lowest_devices = [d[0] for d in queued if d[1] == lowest_count]
        lowest_devices.sort()

        return lowest_devices[0]

    def cancel(self, key: str) -> bool:
        """
        Cancel a job. If the job has not been started, this will cancel
        the future and never execute it. If the job has been started, it
        should be cancelled on the next progress callback.
        """

        for job in self.finished_jobs:
            if job.job == key:
                logger.debug("cannot cancel finished job: %s", key)
                return False

        for job in self.pending_jobs:
            if job.name == key:
                self.pending_jobs.remove(job)
                logger.info("cancelled pending job: %s", key)
                return True

        if key not in self.running_jobs:
            logger.debug("cancelled job is not active: %s", key)
        else:
            job = self.running_jobs[key]
            logger.info("cancelling job %s, active on device %s", key, job.device)

        self.cancelled_jobs.append(key)
        return True

    def done(self, key: str) -> Tuple[bool, Optional[ProgressCommand]]:
        """
        Check if a job has been finished and report the last progress update.

        If the job is still pending, the first item will be True and there will be no ProgressCommand.
        """
        if key in self.running_jobs:
            logger.debug("checking status for running job: %s", key)
            return (False, self.running_jobs[key])

        for job in self.finished_jobs:
            if job.job == key:
                logger.debug("checking status for finished job: %s", key)
                return (False, job)

        for job in self.pending_jobs:
            if job.name == key:
                logger.debug("checking status for pending job: %s", key)
                return (True, None)

        logger.trace("checking status for unknown job: %s", key)
        return (False, None)

    def join(self):
        logger.info("stopping worker pool")

        with self.rlock:
            logger.debug("stopping device workers")
            for device, worker in self.workers.items():
                if worker.is_alive():
                    logger.debug("stopping worker %s for device %s", worker.pid, device)
                    worker.join(self.join_timeout)
                    if worker.is_alive():
                        logger.warning(
                            "worker %s for device %s could not be stopped in time",
                            worker.pid,
                            device,
                        )
                        self.leak_worker(device)
                else:
                    logger.debug("worker for device %s has died", device)

            logger.debug("stopping health worker")
            self.health_worker.cancel()
            self.health_worker.join(self.recycle_interval)

            logger.debug("stopping progress worker")
            self.progress_worker.cancel()
            self.progress_worker.join(self.progress_interval)

            logger.debug("stopping logger worker")
            self.logger_worker.join(self.join_timeout)

            logger.debug("closing worker queues")
            self.logs.close()

            for queue in self.pending.values():
                queue.close()

            for queue in self.progress.values():
                queue.close()

            self.pending.clear()
            self.progress.clear()

            logger.debug("stopping leaking workers")
            self.join_leaking()

            logger.debug("worker pool stopped")

    def join_leaking(self):
        if len(self.leaking) > 0:
            for device, worker, context in self.leaking:
                logger.debug(
                    "shutting down leaking worker %s for device %s", worker.pid, device
                )
                worker.join(self.join_timeout)
                if worker.is_alive():
                    logger.warning(
                        "leaking worker %s for device %s could not be shut down",
                        worker.pid,
                        device,
                    )

                try:
                    progress = context.progress.get_nowait()
                    while progress is not None:
                        self.update_job(progress)
                        progress = context.progress.get_nowait()
                except Empty:
                    logger.trace("empty queue in leaking worker for device %s", device)
                except ValueError as e:
                    logger.debug(
                        "value error in leaking worker for device %s: %s", device, e
                    )
                except Exception:
                    logger.exception("error in leaking worker for device %s", device)

            self.leaking[:] = [dw for dw in self.leaking if dw[1].is_alive()]

    def recycle(self, recycle_all=False):
        logger.debug("recycling worker pool")

        with self.rlock:
            self.join_leaking()

            needs_restart = []

            for device, worker in self.workers.items():
                jobs = self.total_jobs.get(device, 0)
                if not worker.is_alive():
                    logger.warning("worker for device %s has died", device)
                    needs_restart.append(device)
                elif recycle_all or jobs > self.max_jobs_per_worker:
                    logger.info(
                        "shutting down worker for device %s after %s jobs", device, jobs
                    )
                    worker.join(self.join_timeout)
                    if worker.is_alive():
                        logger.warning(
                            "worker %s for device %s could not be shut down in time",
                            worker.pid,
                            device,
                        )
                        self.leak_worker(device)
                    else:
                        del worker

                    self.workers[device] = None
                    needs_restart.append(device)
                else:
                    logger.debug(
                        "worker %s for device %s has run %s jobs and is still alive",
                        worker.pid,
                        device,
                        jobs,
                    )

            if len(needs_restart) > 0:
                logger.info("starting new workers")

            for device in self.devices:
                if device.device in needs_restart:
                    self.create_device_worker(device)

            if self.logger_worker.is_alive():
                logger.debug("logger worker is running")
            else:
                logger.warning("restarting logger worker")
                self.create_logger_worker()

            if self.progress_worker.is_alive():
                logger.debug("progress worker is running")
            else:
                logger.warning("restarting progress worker")
                self.create_progress_worker()

            logger.debug("worker pool recycled")

    def submit(
        self,
        key: str,
        fn: Callable[..., None],
        /,
        *args,
        needs_device: Optional[DeviceParams] = None,
        **kwargs,
    ) -> None:
        device_idx = self.get_next_device(needs_device=needs_device)
        device = self.devices[device_idx].device
        logger.info(
            "assigning job %s to device %s: %s",
            key,
            device_idx,
            device,
        )

        # build and queue job
        job = JobCommand(key, device, fn, args, kwargs)
        self.pending_jobs.append(job)

    def status(self) -> Dict[str, List[Tuple[str, int, bool, bool, bool, bool]]]:
        """
        Returns a tuple of: job/device, progress, progress, finished, cancelled, failed
        """
        return {
            "cancelled": [],
            "finished": [
                (
                    job.job,
                    job.progress,
                    False,
                    job.finished,
                    job.cancelled,
                    job.failed,
                )
                for job in self.finished_jobs
            ],
            "pending": [
                (
                    job.name,
                    0,
                    True,
                    False,
                    False,
                    False,
                )
                for job in self.pending_jobs
            ],
            "running": [
                (
                    name,
                    job.progress,
                    False,
                    job.finished,
                    job.cancelled,
                    job.failed,
                )
                for name, job in self.running_jobs.items()
            ],
            "total": [
                (
                    device,
                    total,
                    self.workers[device].is_alive(),
                    False,
                    False,
                    False,
                )
                for device, total in self.total_jobs.items()
            ],
        }

    def next_job(self, device: str):
        for job in self.pending_jobs:
            if job.device == device:
                logger.debug("enqueuing job %s on device %s", job.name, device)
                self.pending[device].put(job, block=False)
                # job will be removed from pending queue when progress is updated
                return

        logger.trace("no pending jobs for device %s", device)

    def finish_job(self, progress: ProgressCommand):
        # move from running to finished
        logger.info("job has finished: %s", progress.job)
        self.finished_jobs.append(progress)
        if progress.job in self.running_jobs:
            del self.running_jobs[progress.job]

        self.join_leaking()
        if progress.job in self.cancelled_jobs:
            self.cancelled_jobs.remove(progress.job)

    def update_job(self, progress: ProgressCommand):
        if progress.finished:
            return self.finish_job(progress)

        # move from pending to running
        logger.debug(
            "progress update for job: %s to %s", progress.job, progress.progress
        )
        self.running_jobs[progress.job] = progress
        self.pending_jobs[:] = [
            job for job in self.pending_jobs if job.name != progress.job
        ]

        # increment job counter if this is the start of a new job
        if progress.progress == 0:
            if progress.device in self.total_jobs:
                self.total_jobs[progress.device] += 1
            else:
                self.total_jobs[progress.device] = 1

            logger.debug(
                "updating job count for device %s: %s",
                progress.device,
                self.total_jobs[progress.device],
            )

        # check if the job has been cancelled
        if progress.job in self.cancelled_jobs:
            logger.debug(
                "setting flag for cancelled job: %s on %s",
                progress.job,
                progress.device,
            )
            self.context[progress.device].set_cancel()

    def leak_worker(self, device: str):
        context = self.context[device]
        worker = self.workers[device]
        self.leaking.append((device, worker, context))


def health_main(pool: DevicePoolExecutor):
    logger.trace("checking in from health worker thread")
    pool.recycle()

    if pool.logs.full():
        logger.warning("logger queue is full, restarting worker")
        pool.logger_worker.join(pool.join_timeout)
        pool.create_logger_worker()

    if any([queue.full() for queue in pool.progress.values()]):
        logger.warning("progress queue is full, restarting worker")
        pool.progress_worker.join(pool.join_timeout)
        pool.create_progress_worker()


def logger_main(pool: DevicePoolExecutor, logs: "Queue[str]"):
    logger.trace("checking in from logger worker thread")

    while True:
        try:
            msg = logs.get(pool.join_timeout / 2)
            logger.debug("received logs from worker: %s", msg)
        except Empty:
            # logger worker should not generate more logs if it doesn't have any logs
            pass
        except ValueError:
            break
        except Exception:
            logger.exception("error in log worker")


def progress_main(pool: DevicePoolExecutor):
    logger.trace("checking in from progress worker thread")

    for device, _worker, context in pool.leaking:
        # whether the worker is alive or not, try to clear its queues
        try:
            progress = context.progress.get_nowait()
            while progress is not None:
                pool.update_job(progress)
                progress = context.progress.get_nowait()
        except Empty:
            logger.trace("empty queue in leaking worker for device %s", device)
        except ValueError as e:
            logger.debug("value error in leaking worker for device %s: %s", device, e)
            break
        except Exception:
            logger.exception("error in leaking worker for device %s", device)

    for device, queue in pool.progress.items():
        try:
            progress = queue.get_nowait()
            while progress is not None:
                pool.update_job(progress)
                progress = queue.get_nowait()
        except Empty:
            logger.trace("empty queue in progress worker for device %s", device)
        except ValueError as e:
            logger.debug("value error in progress worker for device %s: %s", device, e)
            break
        except Exception:
            logger.exception("error in progress worker for device %s", device)

    for device, context in pool.context.items():
        if context.is_idle():
            logger.trace("enqueueing next job for idle worker")
            pool.next_job(device)
