from typing import Callable, Any

class ProgressCommand():
  device: str
  job: str
  finished: bool
  progress: int
  cancel: bool
  error: bool

  def __init__(
    self,
    job: str,
    device: str,
    finished: bool,
    progress: int,
    cancel: bool = False,
    error: bool = False,
  ):
    self.job = job
    self.device = device
    self.finished = finished
    self.progress = progress
    self.cancel = cancel
    self.error = error

class JobCommand():
  name: str
  fn: Callable[..., None]
  args: Any
  kwargs: dict[str, Any]

  def __init__(
    self,
    name: str,
    fn: Callable[..., None],
    args: Any,
    kwargs: dict[str, Any],
  ):
    self.name = name
    self.fn = fn
    self.args = args
    self.kwargs = kwargs
