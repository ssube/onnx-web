import gc
import threading
from logging import getLogger
from typing import List, Optional

import torch

from .params import DeviceParams

logger = getLogger(__name__)


def run_gc(devices: Optional[List[DeviceParams]] = None):
    logger.debug(
        "running garbage collection with %s active threads", threading.active_count()
    )
    gc.collect()

    if torch.cuda.is_available() and devices is not None:
        for device in [d for d in devices if d.device.startswith("cuda")]:
            logger.debug("running Torch garbage collection for device: %s", device)
            with torch.cuda.device(device.torch_str()):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                mem_free, mem_total = torch.cuda.mem_get_info()
                mem_pct = (1 - (mem_free / mem_total)) * 100
                logger.debug(
                    "CUDA VRAM usage: %s of %s (%.2f%%)",
                    (mem_total - mem_free),
                    mem_total,
                    mem_pct,
                )
