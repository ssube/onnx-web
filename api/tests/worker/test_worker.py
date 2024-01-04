import unittest
from multiprocessing import Queue, Value
from os import getpid

from onnx_web.errors import RetryException
from onnx_web.server.context import ServerContext
from onnx_web.worker.command import JobCommand
from onnx_web.worker.context import WorkerContext
from onnx_web.worker.worker import (
    EXIT_ERROR,
    EXIT_INTERRUPT,
    EXIT_MEMORY,
    EXIT_REPLACED,
    MEMORY_ERRORS,
    worker_main,
)
from tests.helpers import test_device


def main_memory(_worker):
    raise MemoryError(MEMORY_ERRORS[0])


def main_retry(_worker):
    raise RetryException()


def main_interrupt(_worker):
    raise KeyboardInterrupt()


class WorkerMainTests(unittest.TestCase):
    def test_pending_exception_empty(self):
        pass

    def test_pending_exception_interrupt(self):
        status = None

        def exit(exit_status):
            nonlocal status
            status = exit_status

        job = JobCommand("test", "test", "test", main_interrupt, [], {})
        cancel = Value("L", False)
        logs = Queue()
        pending = Queue()
        progress = Queue()
        pid = Value("L", getpid())
        idle = Value("L", False)

        pending.put(job)
        worker_main(
            WorkerContext(
                "test",
                test_device(),
                cancel,
                logs,
                pending,
                progress,
                pid,
                idle,
                0,
                0.0,
            ),
            ServerContext(),
            exit=exit,
        )

        self.assertEqual(status, EXIT_INTERRUPT)

    def test_pending_exception_retry(self):
        status = None

        def exit(exit_status):
            nonlocal status
            status = exit_status

        job = JobCommand("test", "test", "test", main_retry, [], {})
        cancel = Value("L", False)
        logs = Queue()
        pending = Queue()
        progress = Queue()
        pid = Value("L", getpid())
        idle = Value("L", False)

        pending.put(job)
        worker_main(
            WorkerContext(
                "test",
                test_device(),
                cancel,
                logs,
                pending,
                progress,
                pid,
                idle,
                0,
                0.0,
            ),
            ServerContext(),
            exit=exit,
        )

        self.assertEqual(status, EXIT_ERROR)

    def test_pending_exception_value(self):
        status = None

        def exit(exit_status):
            nonlocal status
            status = exit_status

        cancel = Value("L", False)
        logs = Queue()
        pending = Queue()
        progress = Queue()
        pid = Value("L", getpid())
        idle = Value("L", False)

        pending.close()
        worker_main(
            WorkerContext(
                "test",
                test_device(),
                cancel,
                logs,
                pending,
                progress,
                pid,
                idle,
                0,
                0.0,
            ),
            ServerContext(),
            exit=exit,
        )

        self.assertEqual(status, EXIT_ERROR)

    def test_pending_exception_other_memory(self):
        status = None

        def exit(exit_status):
            nonlocal status
            status = exit_status

        job = JobCommand("test", "test", "test", main_memory, [], {})
        cancel = Value("L", False)
        logs = Queue()
        pending = Queue()
        progress = Queue()
        pid = Value("L", getpid())
        idle = Value("L", False)

        pending.put(job)
        worker_main(
            WorkerContext(
                "test",
                test_device(),
                cancel,
                logs,
                pending,
                progress,
                pid,
                idle,
                0,
                0.0,
            ),
            ServerContext(),
            exit=exit,
        )

        self.assertEqual(status, EXIT_MEMORY)

    def test_pending_exception_other_unknown(self):
        pass

    def test_pending_replaced(self):
        status = None

        def exit(exit_status):
            nonlocal status
            status = exit_status

        cancel = Value("L", False)
        logs = Queue()
        pending = Queue()
        progress = Queue()
        pid = Value("L", 0)
        idle = Value("L", False)

        worker_main(
            WorkerContext(
                "test",
                test_device(),
                cancel,
                logs,
                pending,
                progress,
                pid,
                idle,
                0,
                0.0,
            ),
            ServerContext(),
            exit=exit,
        )

        self.assertEqual(status, EXIT_REPLACED)
