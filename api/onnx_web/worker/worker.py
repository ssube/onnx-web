from logging import getLogger
import torch # has to come before ORT
from onnxruntime import get_available_providers
from torch.multiprocessing import Lock, Queue
from traceback import format_exception
from setproctitle import setproctitle

from .context import WorkerContext
from ..server import ServerContext, apply_patches

logger = getLogger(__name__)


def logger_init(lock: Lock, logs: Queue):
    with lock:
        logger.info("checking in from logger, %s", lock)

    setproctitle("onnx-web logger")

    while True:
        job = logs.get()
        with open("worker.log", "w") as f:
            logger.info("got log: %s", job)
            f.write(str(job) + "\n\n")


def worker_init(lock: Lock, context: WorkerContext, server: ServerContext):
    with lock:
        logger.info("checking in from worker, %s, %s", lock, get_available_providers())

    apply_patches(server)
    setproctitle("onnx-web worker: %s", context.device.device)

    while True:
        job = context.pending.get()
        logger.info("got job: %s", job)
        try:
            fn, args, kwargs = job
            name = args[3][0]
            logger.info("starting job: %s", name)
            with context.finished.get_lock():
                context.finished.value = False

            with context.progress.get_lock():
                context.progress.value = 0

            fn(context, *args, **kwargs)
            logger.info("finished job: %s", name)

            with context.finished.get_lock():
                context.finished.value = True

        except Exception as e:
            logger.error(format_exception(type(e), e, e.__traceback__))

