from typing import Any, Callable


class ProgressCommand:
    device: str
    job: str
    finished: bool
    progress: int
    cancelled: bool
    failed: bool

    def __init__(
        self,
        job: str,
        device: str,
        finished: bool,
        progress: int,
        cancelled: bool = False,
        failed: bool = False,
    ):
        self.job = job
        self.device = device
        self.finished = finished
        self.progress = progress
        self.cancelled = cancelled
        self.failed = failed


class JobCommand:
    device: str
    name: str
    fn: Callable[..., None]
    args: Any
    kwargs: dict[str, Any]

    def __init__(
        self,
        name: str,
        device: str,
        fn: Callable[..., None],
        args: Any,
        kwargs: dict[str, Any],
    ):
        self.name = name
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
