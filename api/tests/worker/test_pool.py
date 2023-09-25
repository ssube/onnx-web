import unittest
from multiprocessing import Event
from time import sleep
from typing import Optional

from onnx_web.params import DeviceParams
from onnx_web.server.context import ServerContext
from onnx_web.worker.pool import DevicePoolExecutor

TEST_JOIN_TIMEOUT = 0.2

def test_job(*args, lock: Event, **kwargs):
  lock.wait()


class TestWorkerPool(unittest.TestCase):
  lock: Optional[Event]
  pool: Optional[DevicePoolExecutor]

  def setUp(self) -> None:
    self.lock = Event()
    self.pool = None

  def tearDown(self) -> None:
    if self.pool is not None:
        self.pool.join()

  def test_no_devices(self):
    server = ServerContext()
    self.pool = DevicePoolExecutor(server, [], join_timeout=TEST_JOIN_TIMEOUT)
    self.pool.start()

  def test_fake_worker(self):
    device = DeviceParams("cpu", "CPUProvider")
    server = ServerContext()
    self.pool = DevicePoolExecutor(server, [device], join_timeout=TEST_JOIN_TIMEOUT)
    self.pool.start()
    self.assertEqual(len(self.pool.workers), 1)

  def test_cancel_pending(self):
    pass

  def test_cancel_running(self):
    pass

  def test_next_device(self):
    device = DeviceParams("cpu", "CPUProvider")
    server = ServerContext()
    self.pool = DevicePoolExecutor(server, [device], join_timeout=TEST_JOIN_TIMEOUT)
    self.pool.start()

    self.assertEqual(self.pool.get_next_device(), 0)

  def test_needs_device(self):
    device1 = DeviceParams("cpu1", "CPUProvider")
    device2 = DeviceParams("cpu2", "CPUProvider")
    server = ServerContext()
    self.pool = DevicePoolExecutor(server, [device1, device2], join_timeout=TEST_JOIN_TIMEOUT)
    self.pool.start()

    self.assertEqual(self.pool.get_next_device(needs_device=device2), 1)

  def test_done_running(self):
    """
    device = DeviceParams("cpu", "CPUProvider")
    server = ServerContext()

    self.pool = DevicePoolExecutor(server, [device], join_timeout=TEST_JOIN_TIMEOUT)
    self.pool.start()

    self.pool.submit("test", test_job, lock=self.lock)
    sleep(5.0)
    self.assertEqual(self.pool.done("test"), (False, None))
    """
    pass

  def test_done_pending(self):
    device = DeviceParams("cpu", "CPUProvider")
    server = ServerContext()

    self.pool = DevicePoolExecutor(server, [device], join_timeout=TEST_JOIN_TIMEOUT)
    self.pool.start()

    self.pool.submit("test1", test_job, lock=self.lock)
    self.pool.submit("test2", test_job, lock=self.lock)
    self.assertTrue(self.pool.done("test2"), (True, None))

    self.lock.set()

  def test_done_finished(self):
    """
    device = DeviceParams("cpu", "CPUProvider")
    server = ServerContext()

    self.pool = DevicePoolExecutor(server, [device], join_timeout=TEST_JOIN_TIMEOUT)
    self.pool.start()

    self.pool.submit("test", test_job, lock=self.lock)
    self.assertEqual(self.pool.done("test"), (True, None))

    self.lock.set()
    sleep(5.0)
    self.assertEqual(self.pool.done("test"), (False, None))
    """
    pass

  def test_recycle_live(self):
    pass

  def test_recycle_dead(self):
    pass

  def test_running_status(self):
    pass