from logging import getLogger
from os import getpid
from queue import Empty
from sys import exit

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

    logger.trace("checking in from worker, %s", get_available_providers())

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

            job = context.pending.get(timeout=1.0)
            logger.info("worker for %s got job: %s", context.device.device, job.name)

            context.job = job.name  # TODO: hax
            logger.info("starting job: %s", job.name)
            context.set_progress(0)
            job.fn(context, *job.args, **job.kwargs)
            logger.info("job succeeded: %s", job.name)
            context.set_finished()
        except Empty:
            pass
        except KeyboardInterrupt:
            logger.info("worker got keyboard interrupt")
            context.set_failed()
            exit(EXIT_INTERRUPT)
        except ValueError:
            logger.exception("value error in worker, exiting: %s")
            context.set_failed()
            exit(EXIT_ERROR)
        except Exception as e:
            e_str = str(e)
            if "Failed to allocate memory" in e_str or "out of memory" in e_str:
                logger.error("detected out-of-memory error, exiting: %s", e)
                context.set_failed()
                exit(EXIT_MEMORY)
            else:
                logger.exception(
                    "error while running job",
                )
                context.set_failed()
                # carry on
