from logging import getLogger
from os import getpid
from queue import Empty
from sys import exit

from setproctitle import setproctitle

from ..errors import RetryException
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


def worker_main(worker: WorkerContext, server: ServerContext, *args):
    apply_patches(server)
    setproctitle("onnx-web worker: %s" % (worker.device.device))

    logger.trace(
        "checking in from worker with providers: %s", get_available_providers()
    )

    # make leaking workers easier to recycle
    worker.progress.cancel_join_thread()

    while True:
        try:
            if not worker.is_active():
                logger.warning(
                    "worker %s has been replaced by %s, exiting",
                    getpid(),
                    worker.get_active(),
                )
                exit(EXIT_REPLACED)

            # wait briefly for the next job
            job = worker.pending.get(timeout=worker.timeout)
            logger.info("worker %s got job: %s", worker.device.device, job.name)

            # clear flags and save the job name
            worker.start(job.name)
            logger.info("starting job: %s", job.name)

            # reset progress, which does a final check for cancellation
            worker.set_progress(0)
            job.fn(worker, *job.args, **job.kwargs)

            # confirm completion of the job
            logger.info("job succeeded: %s", job.name)
            worker.finish()
        except Empty:
            logger.trace("worker reached end of queue, setting idle flag")
            worker.set_idle()
        except KeyboardInterrupt:
            logger.debug("worker got keyboard interrupt")
            worker.fail()
            exit(EXIT_INTERRUPT)
        except RetryException:
            logger.exception("retry error in worker, exiting")
            worker.fail()
            exit(EXIT_ERROR)
        except ValueError:
            logger.exception("value error in worker, exiting")
            worker.fail()
            exit(EXIT_ERROR)
        except Exception as e:
            e_str = str(e)
            # restart the worker on memory errors
            for e_mem in MEMORY_ERRORS:
                if e_mem in e_str:
                    logger.error("detected out-of-memory error, exiting: %s", e)
                    worker.fail()
                    exit(EXIT_MEMORY)

            # carry on for other errors
            logger.exception(
                "unrecognized error while running job",
            )
            worker.fail()
