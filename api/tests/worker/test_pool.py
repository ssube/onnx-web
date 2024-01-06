import unittest
from multiprocessing import Event
from time import sleep
from typing import Optional

from onnx_web.params import DeviceParams
from onnx_web.server.context import ServerContext
from onnx_web.worker.command import JobStatus
from onnx_web.worker.pool import DevicePoolExecutor
from tests.helpers import test_device

TEST_JOIN_TIMEOUT = 0.2

lock = Event()


def lock_job(*args, **kwargs):
    lock.wait()


def sleep_job(*args, **kwargs):
    sleep(0.5)


def progress_job(worker, *args, **kwargs):
    worker.set_progress(1)


def fail_job(*args, **kwargs):
    raise RuntimeError("job failed")


class TestWorkerPool(unittest.TestCase):
    # lock: Optional[Event]
    pool: Optional[DevicePoolExecutor]

    def setUp(self) -> None:
        self.pool = None

    def tearDown(self) -> None:
        if self.pool is not None:
            self.pool.join()

    def test_no_devices(self):
        server = ServerContext()
        self.pool = DevicePoolExecutor(server, [], join_timeout=TEST_JOIN_TIMEOUT)
        self.pool.start()

    def test_fake_worker(self):
        device = test_device()
        server = ServerContext()
        self.pool = DevicePoolExecutor(server, [device], join_timeout=TEST_JOIN_TIMEOUT)
        self.pool.start()
        self.assertEqual(len(self.pool.workers), 1)

    def test_cancel_pending(self):
        device = test_device()
        server = ServerContext()

        self.pool = DevicePoolExecutor(server, [device], join_timeout=TEST_JOIN_TIMEOUT)
        self.pool.start()

        self.pool.submit("test", "test", sleep_job, lock=lock)
        self.assertEqual(self.pool.status("test"), (JobStatus.PENDING, None))

        self.assertTrue(self.pool.cancel("test"))
        self.assertEqual(self.pool.status("test"), (JobStatus.CANCELLED, None))

    def test_cancel_running(self):
        pass

    def test_next_device(self):
        device = test_device()
        server = ServerContext()
        self.pool = DevicePoolExecutor(server, [device], join_timeout=TEST_JOIN_TIMEOUT)
        self.pool.start()

        self.assertEqual(self.pool.get_next_device(), 0)

    def test_needs_device(self):
        device1 = DeviceParams("cpu1", "CPUProvider")
        device2 = DeviceParams("cpu2", "CPUProvider")
        server = ServerContext()
        self.pool = DevicePoolExecutor(
            server, [device1, device2], join_timeout=TEST_JOIN_TIMEOUT
        )
        self.pool.start()

        self.assertEqual(self.pool.get_next_device(needs_device=device2), 1)

    def test_done_running(self):
        """
        TODO: flaky
        """
        device = test_device()
        server = ServerContext()

        self.pool = DevicePoolExecutor(
            server, [device], join_timeout=TEST_JOIN_TIMEOUT, progress_interval=0.1
        )

        lock.clear()
        self.pool.start(lock)
        self.pool.submit("test", "test", lock_job)
        sleep(5.0)

        status, _progress = self.pool.status("test")
        self.assertEqual(status, JobStatus.RUNNING)

    def test_done_pending(self):
        device = test_device()
        server = ServerContext()

        self.pool = DevicePoolExecutor(server, [device], join_timeout=TEST_JOIN_TIMEOUT)
        self.pool.start(lock)

        self.pool.submit("test1", "test", lock_job)
        self.pool.submit("test2", "test", lock_job)
        self.assertEqual(self.pool.status("test2"), (JobStatus.PENDING, None))

        lock.set()

    def test_done_finished(self):
        """
        TODO: flaky
        """
        device = test_device()
        server = ServerContext()

        self.pool = DevicePoolExecutor(
            server, [device], join_timeout=TEST_JOIN_TIMEOUT, progress_interval=0.1
        )
        self.pool.start()
        self.pool.submit("test", "test", sleep_job)
        self.assertEqual(self.pool.status("test"), (JobStatus.PENDING, None))

        sleep(5.0)
        status, _progress = self.pool.status("test")
        self.assertEqual(status, JobStatus.SUCCESS)

    def test_recycle_live(self):
        pass

    def test_recycle_dead(self):
        pass

    def test_running_status(self):
        pass

    def test_progress_update(self):
        pass

    def test_progress_finished(self):
        device = test_device()
        server = ServerContext()

        self.pool = DevicePoolExecutor(
            server, [device], join_timeout=TEST_JOIN_TIMEOUT, progress_interval=0.1
        )
        self.pool.start()

        self.pool.submit("test", "test", progress_job)
        sleep(5.0)

        status, progress = self.pool.status("test")
        self.assertEqual(status, JobStatus.SUCCESS)
        self.assertEqual(progress.steps.current, 1)

    def test_progress_failed(self):
        device = test_device()
        server = ServerContext()

        self.pool = DevicePoolExecutor(
            server, [device], join_timeout=TEST_JOIN_TIMEOUT, progress_interval=0.1
        )
        self.pool.start()

        self.pool.submit("test", "test", fail_job)
        sleep(5.0)

        status, progress = self.pool.status("test")
        self.assertEqual(status, JobStatus.FAILED)
        self.assertEqual(progress.steps.current, 0)
