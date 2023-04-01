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

MEMORY_ERRORS = [
    "Failed to allocate memory",
    "hipErrorOutOfMemory",
    "MIOPEN failure",
    "out of memory",
    "rocblas_status_memory_error",
]


def worker_main(context: WorkerContext, server: ServerContext):
    apply_patches(server)
    setproctitle("onnx-web worker: %s" % (context.device.device))

    logger.trace(
        "checking in from worker with providers: %s", get_available_providers()
    )

    # make leaking workers easier to recycle
    context.progress.cancel_join_thread()

    while True:
        try:
            if not context.is_active():
                logger.warning(
                    "worker %s has been replaced by %s, exiting",
                    getpid(),
                    context.get_active(),
                )
                exit(EXIT_REPLACED)

            # wait briefly for the next job
            job = context.pending.get(timeout=1.0)
            logger.info("worker %s got job: %s", context.device.device, job.name)

            # clear flags and save the job name
            context.start(job.name)
            logger.info("starting job: %s", job.name)

            # reset progress, which does a final check for cancellation
            context.set_progress(0)
            job.fn(context, *job.args, **job.kwargs)

            # confirm completion of the job
            logger.info("job succeeded: %s", job.name)
            context.finish()
        except Empty:
            pass
        except KeyboardInterrupt:
            logger.info("worker got keyboard interrupt")
            context.fail()
            exit(EXIT_INTERRUPT)
        except ValueError:
            logger.exception("value error in worker, exiting: %s")
            context.fail()
            exit(EXIT_ERROR)
        except Exception as e:
            e_str = str(e)
            # restart the worker on memory errors
            for e_mem in MEMORY_ERRORS:
                if e_mem in e_str:
                    logger.error("detected out-of-memory error, exiting: %s", e)
                    context.fail()
                    exit(EXIT_MEMORY)

            # carry on for other errors
            logger.exception(
                "unrecognized error while running job",
            )
            context.fail()
