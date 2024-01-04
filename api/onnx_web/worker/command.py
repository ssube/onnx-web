from enum import Enum
from typing import Any, Callable, Dict


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


class JobType(str, Enum):
    TXT2TXT = "txt2txt"
    TXT2IMG = "txt2img"
    IMG2IMG = "img2img"
    INPAINT = "inpaint"
    UPSCALE = "upscale"
    BLEND = "blend"
    CHAIN = "chain"


class ProgressCommand:
    device: str
    job: str
    job_type: str
    status: JobStatus
    results: int
    steps: int
    stages: int
    tiles: int

    def __init__(
        self,
        job: str,
        job_type: str,
        device: str,
        status: JobStatus,
        results: int = 0,
        steps: int = 0,
        stages: int = 0,
        tiles: int = 0,
    ):
        self.job = job
        self.job_type = job_type
        self.device = device
        self.status = status
        self.results = results
        self.steps = steps
        self.stages = stages
        self.tiles = tiles


class JobCommand:
    device: str
    name: str
    job_type: str
    fn: Callable[..., None]
    args: Any
    kwargs: Dict[str, Any]

    def __init__(
        self,
        name: str,
        device: str,
        job_type: str,
        fn: Callable[..., None],
        args: Any,
        kwargs: Dict[str, Any],
    ):
        self.device = device
        self.name = name
        self.job_type = job_type
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
