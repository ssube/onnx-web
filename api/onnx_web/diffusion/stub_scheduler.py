from typing import Any


class StubScheduler:
    def step(
        self, model_output: Any, timestep: int, sample: Any, return_dict: bool = True
    ) -> None:
        raise NotImplementedError("scheduler not available, try updating diffusers")
