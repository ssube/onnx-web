import unittest
from multiprocessing import Queue, Value

from onnx_web.server.context import ServerContext
from onnx_web.worker.context import WorkerContext
from onnx_web.worker.worker import EXIT_INTERRUPT, worker_main
from tests.helpers import test_device


class WorkerMainTests(unittest.TestCase):
  def test_pending_exception_empty(self):
    pass

  def test_pending_exception_interrupt(self):
    status = None

    def exit(exit_status):
      status = exit_status

    cancel = Value("L", False)
    logs = Queue()
    pending = Queue()
    progress = Queue()
    pid = Value("L", False)
    idle = Value("L", False)

    pending.close()
    # worker_main(WorkerContext("test", test_device(), cancel, logs, pending, progress, pid, idle, 0, 0.0), ServerContext(), exit=exit)

    self.assertEqual(status, EXIT_INTERRUPT)

  def test_pending_exception_retry(self):
    pass

  def test_pending_exception_value(self):
    pass

  def test_pending_exception_other_memory(self):
    pass

  def test_pending_exception_other_unknown(self):
    pass
