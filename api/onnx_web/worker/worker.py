from logging import getLogger
from os import getpid
from queue import Empty
from sys import exit
from traceback import format_exception

from setproctitle import setproctitle

from ..server import ServerContext, apply_patches
from ..torch_before_ort import get_available_providers
from .context import WorkerContext

logger = getLogger(__name__)

EXIT_ERROR = 1
EXIT_INTERRUPT = 0
EXIT_MEMORY = 2
EXIT_REPLACED = 3
EXIT_SUCCESS = 0


def worker_main(context: WorkerContext, server: ServerContext):
    apply_patches(server)
    setproctitle("onnx-web worker: %s" % (context.device.device))

    logger.info("checking in from worker, %s", get_available_providers())

    # make leaking workers easier to recycle
    context.progress.cancel_join_thread()
    context.finished.cancel_join_thread()

    while True:
        try:
            if not context.is_current():
                logger.warning(
                    "worker %s has been replaced by %s, exiting",
                    getpid(),
                    context.get_current(),
                )
                exit(EXIT_REPLACED)

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
            exit(EXIT_INTERRUPT)
        except ValueError as e:
            logger.info(
                "value error in worker, exiting: %s",
                format_exception(type(e), e, e.__traceback__),
            )
            exit(EXIT_ERROR)
        except Exception as e:
            if "Failed to allocate memory" in str(e):
                logger.error("detected out-of-memory error, exiting: %s", e)
                exit(EXIT_MEMORY)
            else:
                logger.error(
                    "error while running job: %s",
                    format_exception(type(e), e, e.__traceback__),
                )
