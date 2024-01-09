from enum import Enum
from typing import Any, Callable, Dict, Optional


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
    """
    Generic counter with current and expected/final/total value. Can be used to count up or down.

    Counter is considered "complete" when the current value is greater than or equal to the total value, and "empty"
    when the current value is zero.
    """

    current: int
    total: int

    def __init__(self, current: int, total: int) -> None:
        self.current = current
        self.total = total

    def __str__(self) -> str:
        return "%s/%s" % (self.current, self.total)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Progress):
            return self.current == other.current and self.total == other.total

        return False

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
    reason: Optional[str]
    result: Optional[Any]  # really StageResult but that would be a very circular import
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
        reason: Optional[str] = None,
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
        self.reason = reason


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
