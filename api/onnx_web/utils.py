from os import path
from typing import Any, Dict, Tuple


Point = Tuple[int, int]


def get_and_clamp_float(args, key: str, default_value: float, max_value: float, min_value=0.0) -> float:
    return min(max(float(args.get(key, default_value)), min_value), max_value)


def get_and_clamp_int(args, key: str, default_value: int, max_value: int, min_value=1) -> int:
    return min(max(int(args.get(key, default_value)), min_value), max_value)


def get_from_map(args, key: str, values: Dict[str, Any], default: Any):
    selected = args.get(key, default)
    if selected in values:
        return values[selected]
    else:
        return values[default]


def safer_join(base, tail):
    safer_path = path.relpath(path.normpath(path.join('/', tail)), '/')
    return path.join(base, safer_path)


# TODO: .path is only used in one place, can probably just be a str
class OutputPath:
    def __init__(self, path, file):
        self.path = path
        self.file = file


class BaseParams:
    def __init__(self, model, provider, scheduler, prompt, negative_prompt, cfg, steps, seed):
        self.model = model
        self.provider = provider
        self.scheduler = scheduler
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.cfg = cfg
        self.steps = steps
        self.seed = seed

    def tojson(self) -> Dict[str, Any]:
        return {
            'model': self.model,
            'provider': self.provider,
            'scheduler': self.scheduler.__name__,
            'seed': self.seed,
            'prompt': self.prompt,
            'cfg': self.cfg,
            'negativePrompt': self.negative_prompt,
            'steps': self.steps,
        }


class Border:
    def __init__(self, left, right, top, bottom):
        self.left = left
        self.right = right
        self.top = top
        self.bottom = bottom


class Size:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def tojson(self) -> Dict[str, Any]:
        return {
            'height': self.height,
            'width': self.width,
        }
