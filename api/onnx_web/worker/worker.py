from logging import getLogger
from traceback import format_exception

from setproctitle import setproctitle
from torch.multiprocessing import Queue

from ..server import ServerContext, apply_patches
from ..torch_before_ort import get_available_providers
from .context import WorkerContext

logger = getLogger(__name__)


def worker_main(context: WorkerContext, server: ServerContext):
    apply_patches(server)
    setproctitle("onnx-web worker: %s" % (context.device.device))

    logger.info("checking in from worker, %s", get_available_providers())

    while True:
        job = context.pending.get()
        logger.info("got job: %s", job)

        fn, args, kwargs = job
        name = args[3][0]

        try:
            context.job = name  # TODO: hax
            context.clear_flags()
            logger.info("starting job: %s", name)
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
