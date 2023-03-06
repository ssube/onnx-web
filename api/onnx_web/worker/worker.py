from logging import getLogger
from queue import Empty
from sys import exit
from traceback import format_exception

from setproctitle import setproctitle

from ..server import ServerContext, apply_patches
from ..torch_before_ort import get_available_providers
from .context import WorkerContext

logger = getLogger(__name__)


def worker_main(context: WorkerContext, server: ServerContext):
    apply_patches(server)
    setproctitle("onnx-web worker: %s" % (context.device.device))

    logger.info("checking in from worker, %s", get_available_providers())

    # make leaking workers easier to recycle
    context.progress.cancel_join_thread()
    context.finished.cancel_join_thread()

    while True:
        try:
            name, fn, args, kwargs = context.pending.get(timeout=1.0)
            logger.info("worker for %s got job: %s", context.device.device, name)

            context.job = name  # TODO: hax
            context.clear_flags()
            logger.info("starting job: %s", name)
            fn(context, *args, **kwargs)
            logger.info("job succeeded: %s", name)
            context.set_finished()
        except Empty:
            pass
        except KeyboardInterrupt:
            logger.info("worker got keyboard interrupt")
            exit(0)
        except ValueError as e:
            logger.info("value error in worker: %s", e)
            exit(1)
        except Exception as e:
            logger.error(
                "error while running job: %s",
                format_exception(type(e), e, e.__traceback__),
            )
