import unittest

from onnx_web.server.context import ServerContext
from onnx_web.worker.pool import DevicePoolExecutor


class TestWorkerPool(unittest.TestCase):
  def test_no_devices(self):
    server = ServerContext()
    pool = DevicePoolExecutor(server, [])
    pool.start()
    pool.join()