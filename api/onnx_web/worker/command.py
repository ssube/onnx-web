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


class Progress:
    current: int
    total: int

    def __init__(self, current: int, total: int) -> None:
        self.current = current
        self.total = total

    def __str__(self) -> str:
        return "%s/%s" % (self.current, self.total)

    def tojson(self):
        return {
            "current": self.current,
            "total": self.total,
        }

    def complete(self) -> bool:
        return self.current >= self.total

    def empty(self) -> bool:
        # TODO: what if total is also 0?
        return self.current == 0


class ProgressCommand:
    device: str
    job: str
    job_type: str
    status: JobStatus
    result: Any  # really StageResult but that would be a very circular import
    steps: Progress
    stages: Progress
    tiles: Progress

    def __init__(
        self,
        job: str,
        job_type: str,
        device: str,
        status: JobStatus,
        steps: Progress,
        stages: Progress,
        tiles: Progress,
        result: Any = None,
    ):
        self.job = job
        self.job_type = job_type
        self.device = device
        self.status = status

        # progress info
        self.steps = steps
        self.stages = stages
        self.tiles = tiles
        self.result = result


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
