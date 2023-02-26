from logging import getLogger
from onnxruntime import get_available_providers
from torch.multiprocessing import Lock, Queue
from traceback import print_exception

from .context import WorkerContext

logger = getLogger(__name__)

def logger_init(lock: Lock, logs: Queue):
    with lock:
        logger.info("checking in from logger, %s", lock)

    while True:
        job = logs.get()
        with open("worker.log", "w") as f:
            logger.info("got log: %s", job)
            f.write(str(job) + "\n\n")


def worker_init(lock: Lock, context: WorkerContext):
    with lock:
        logger.info("checking in from worker, %s, %s", lock, get_available_providers())

    while True:
        job = context.pending.get()
        logger.info("got job: %s", job)
        try:
            fn, args, kwargs = job
            fn(context, *args, **kwargs)
            logger.info("finished job")
        except Exception as e:
            print_exception(type(e), e, e.__traceback__)

