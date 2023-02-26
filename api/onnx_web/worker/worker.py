from logging import getLogger
from traceback import format_exception

from setproctitle import setproctitle
from torch.multiprocessing import Queue

from ..onnx.torch_before_ort import get_available_providers
from ..server import ServerContext, apply_patches
from .context import WorkerContext

logger = getLogger(__name__)


def logger_init(logs: Queue):
    setproctitle("onnx-web logger")

    logger.info("checking in from logger, %s")

    while True:
        job = logs.get()
        with open("worker.log", "w") as f:
            logger.info("got log: %s", job)
            f.write(str(job) + "\n\n")


def worker_init(context: WorkerContext, server: ServerContext):
    apply_patches(server)
    setproctitle("onnx-web worker: %s" % (context.device.device))

    logger.info("checking in from worker, %s, %s", get_available_providers())

    while True:
        job = context.pending.get()
        logger.info("got job: %s", job)
        try:
            fn, args, kwargs = job
            name = args[3][0]

            logger.info("starting job: %s", name)
            context.clear_flags()
            fn(context, *args, **kwargs)
            logger.info("job succeeded: %s", name)
        except Exception as e:
            logger.error(
                "error while running job: %s",
                format_exception(type(e), e, e.__traceback__),
            )
        finally:
            context.set_finished()
            logger.info("finished job: %s", name)
