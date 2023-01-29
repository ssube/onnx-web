from hashlib import sha256
from logging import getLogger
from os import environ, path
from struct import pack
from time import time
from typing import Any, Dict, List, Optional, Union, Tuple

from .params import (
    ImageParams,
    Param,
    Size,
    SizeChart,
)

logger = getLogger(__name__)


class ServerContext:
    def __init__(
        self,
        bundle_path: str = '.',
        model_path: str = '.',
        output_path: str = '.',
        params_path: str = '.',
        cors_origin: str = '*',
        num_workers: int = 1,
        block_platforms: List[str] = [],
    ) -> None:
        self.bundle_path = bundle_path
        self.model_path = model_path
        self.output_path = output_path
        self.params_path = params_path
        self.cors_origin = cors_origin
        self.num_workers = num_workers
        self.block_platforms = block_platforms

    @classmethod
    def from_environ(cls):
        return ServerContext(
            bundle_path=environ.get('ONNX_WEB_BUNDLE_PATH',
                                    path.join('..', 'gui', 'out')),
            model_path=environ.get('ONNX_WEB_MODEL_PATH',
                                   path.join('..', 'models')),
            output_path=environ.get(
                'ONNX_WEB_OUTPUT_PATH', path.join('..', 'outputs')),
            params_path=environ.get('ONNX_WEB_PARAMS_PATH', '.'),
            # others
            cors_origin=environ.get('ONNX_WEB_CORS_ORIGIN', '*').split(','),
            num_workers=int(environ.get('ONNX_WEB_NUM_WORKERS', 1)),
            block_platforms=environ.get(
                'ONNX_WEB_BLOCK_PLATFORMS', '').split(',')
        )


def is_debug() -> bool:
    return environ.get('DEBUG') is not None


def get_and_clamp_float(args: Any, key: str, default_value: float, max_value: float, min_value=0.0) -> float:
    return min(max(float(args.get(key, default_value)), min_value), max_value)


def get_and_clamp_int(args: Any, key: str, default_value: int, max_value: int, min_value=1) -> int:
    return min(max(int(args.get(key, default_value)), min_value), max_value)


def get_from_list(args: Any, key: str, values: List[Any]) -> Optional[Any]:
    selected = args.get(key, None)
    if selected in values:
        return selected

    logger.warn('invalid selection: %s', selected)
    if len(values) > 0:
        return values[0]

    return None


def get_from_map(args: Any, key: str, values: Dict[str, Any], default: Any) -> Any:
    selected = args.get(key, default)
    if selected in values:
        return values[selected]
    else:
        return values[default]


def get_not_empty(args: Any, key: str, default: Any) -> Any:
    val = args.get(key, default)

    if val is None or len(val) == 0:
        val = default

    return val


def get_size(val: Union[int, str, None]) -> SizeChart:
    if val is None:
        return SizeChart.auto

    if type(val) is str:
        if val in SizeChart:
            return SizeChart[val]
        else:
            return int(val)

    if type(val) is int:
        return val

    raise Exception('invalid size')


def hash_value(sha, param: Param):
    if param is None:
        return
    elif isinstance(param, float):
        sha.update(bytearray(pack('!f', param)))
    elif isinstance(param, int):
        sha.update(bytearray(pack('!I', param)))
    elif isinstance(param, str):
        sha.update(param.encode('utf-8'))
    else:
        logger.warn('cannot hash param: %s, %s', param, type(param))


def make_output_name(
    mode: str,
    params: ImageParams,
    size: Size,
    extras: Optional[Tuple[Param]] = None
) -> str:
    now = int(time())
    sha = sha256()

    hash_value(sha, mode)
    hash_value(sha, params.model)
    hash_value(sha, params.provider)
    hash_value(sha, params.scheduler.__name__)
    hash_value(sha, params.prompt)
    hash_value(sha, params.negative_prompt)
    hash_value(sha, params.cfg)
    hash_value(sha, params.steps)
    hash_value(sha, params.seed)
    hash_value(sha, size.width)
    hash_value(sha, size.height)

    if extras is not None:
        for param in extras:
            hash_value(sha, param)

    return '%s_%s_%s_%s.png' % (mode, params.seed, sha.hexdigest(), now)


def base_join(base: str, tail: str) -> str:
    tail_path = path.relpath(path.normpath(path.join('/', tail)), '/')
    return path.join(base, tail_path)
