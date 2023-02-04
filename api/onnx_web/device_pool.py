from concurrent.futures import Future, ThreadPoolExecutor, ProcessPoolExecutor
from logging import getLogger
from multiprocessing import Value
from typing import Any, Callable, List, Union, Optional

logger = getLogger(__name__)


class JobContext:
    def __init__(
        self,
        key: str,
        devices: List[str],
        cancel: bool = False,
        device_index: int = -1,
        progress: int = 0,
    ):
        self.key = key
        self.devices = list(devices)
        self.cancel = Value('B', cancel)
        self.device_index = Value('i', device_index)
        self.progress = Value('I', progress)

    def is_cancelled(self) -> bool:
        return self.cancel.value

    def get_device(self) -> str:
        '''
        Get the device assigned to this job.
        '''
        with self.device_index.get_lock():
            device_index = self.device_index.value
            if device_index < 0:
                raise Exception('job has not been assigned to a device')
            else:
                return self.devices[device_index]

    def get_progress_callback(self) -> Callable[..., None]:
        def on_progress(step: int, timestep: int, latents: Any):
            if self.is_cancelled():
                raise Exception('job has been cancelled')
            else:
                self.set_progress(step)

        return on_progress

    def set_cancel(self, cancel: bool = True) -> None:
        with self.cancel.get_lock():
            self.cancel.value = cancel

    def set_progress(self, progress: int) -> None:
        with self.progress.get_lock():
            self.progress.value = progress


class Job:
    def __init__(
        self,
        key: str,
        future: Future,
        context: JobContext,
    ):
        self.context = context
        self.future = future
        self.key = key

    def set_cancel(self, cancel: bool = True):
        self.context.set_cancel(cancel)

    def set_progress(self, progress: int):
        self.context.set_progress(progress)


class DevicePoolExecutor:
    devices: List[str] = None
    jobs: List[Job] = None
    pool: Union[ProcessPoolExecutor, ThreadPoolExecutor] = None

    def __init__(self, devices: List[str], pool: Optional[Union[ProcessPoolExecutor, ThreadPoolExecutor]] = None):
        self.devices = devices
        self.jobs = []
        self.pool = pool or ThreadPoolExecutor(len(devices))

    def cancel(self, key: str) -> bool:
        '''
        Cancel a job. If the job has not been started, this will cancel
        the future and never execute it. If the job has been started, it
        should be cancelled on the next progress callback.
        '''
        for job in self.jobs:
            if job.key == key:
                if job.future.cancel():
                    return True
                else:
                    with job.cancel.get_lock():
                        job.cancel.value = True

    def done(self, key: str) -> bool:
        for job in self.jobs:
            if job.key == key:
                return job.future.done()

        logger.warn('checking status for unknown key: %s', key)
        return None

    def prune(self):
        self.jobs[:] = [job for job in self.jobs if job.future.done()]

    def submit(self, key: str, fn: Callable[..., None], /, *args, **kwargs) -> None:
        context = JobContext(key, self.devices, device_index=0)
        future = self.pool.submit(fn, context, *args, **kwargs)
        job = Job(key, future, context)
        self.jobs.append(job)
