from os import path
import time
from struct import pack
from typing import Any, Dict, Tuple, Union
from hashlib import sha256


Param = Union[str, int, float]
Point = Tuple[int, int]


class OutputPath:
    '''
    TODO: .path is only used in one place, can probably just be a str
    '''

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

    def tojson(self) -> Dict[str, Param]:
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
    def __init__(self, left: int, right: int, top: int, bottom: int):
        self.left = left
        self.right = right
        self.top = top
        self.bottom = bottom


class Size:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height

    def tojson(self) -> Dict[str, int]:
        return {
            'height': self.height,
            'width': self.width,
        }


def get_and_clamp_float(args: Any, key: str, default_value: float, max_value: float, min_value=0.0) -> float:
    return min(max(float(args.get(key, default_value)), min_value), max_value)


def get_and_clamp_int(args: Any, key: str, default_value: int, max_value: int, min_value=1) -> int:
    return min(max(int(args.get(key, default_value)), min_value), max_value)


def get_from_map(args: Any, key: str, values: Dict[str, Any], default: Any):
    selected = args.get(key, default)
    if selected in values:
        return values[selected]
    else:
        return values[default]


def safer_join(base: str, tail: str) -> str:
    safer_path = path.relpath(path.normpath(path.join('/', tail)), '/')
    return path.join(base, safer_path)


def hash_value(sha, param: Param):
    '''
    TODO: include functions by name
    '''
    if param is None:
        return
    elif isinstance(param, float):
        sha.update(bytearray(pack('!f', param)))
    elif isinstance(param, int):
        sha.update(bytearray(pack('!I', param)))
    elif isinstance(param, str):
        sha.update(param.encode('utf-8'))
    else:
        print('cannot hash param: %s, %s' % (param, type(param)))


def make_output_path(
    root: str,
    mode: str,
    params: BaseParams,
    size: Size,
    extras: Union[None, Tuple[Param]] = None
) -> OutputPath:
    now = int(time.time())
    sha = sha256()

    hash_value(mode)
    hash_value(params.model)
    hash_value(params.provider)
    hash_value(params.scheduler.__name__)
    hash_value(params.prompt)
    hash_value(params.negative_prompt)
    hash_value(params.cfg)
    hash_value(params.steps)
    hash_value(params.seed)
    hash_value(size.width)
    hash_value(size.height)

    if extras is not None:
        for param in extras:
            hash_value(sha, param)

    output_file = '%s_%s_%s_%s.png' % (mode, params.seed, sha.hexdigest(), now)
    output_full = safer_join(root, output_file)

    return OutputPath(output_full, output_file)
