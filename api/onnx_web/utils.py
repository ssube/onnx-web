import gc
from logging import getLogger
from os import environ, path
from typing import Any, Dict, List, Optional, Union

import torch

from .params import SizeChart
from .server.model_cache import ModelCache

logger = getLogger(__name__)


class ServerContext:
    def __init__(
        self,
        bundle_path: str = ".",
        model_path: str = ".",
        output_path: str = ".",
        params_path: str = ".",
        cors_origin: str = "*",
        num_workers: int = 1,
        any_platform: bool = True,
        block_platforms: List[str] = [],
        default_platform: str = None,
        image_format: str = "png",
        cache: ModelCache = None,
        cache_path: str = None,
    ) -> None:
        self.bundle_path = bundle_path
        self.model_path = model_path
        self.output_path = output_path
        self.params_path = params_path
        self.cors_origin = cors_origin
        self.num_workers = num_workers
        self.any_platform = any_platform
        self.block_platforms = block_platforms
        self.default_platform = default_platform
        self.image_format = image_format
        self.cache = cache or ModelCache(num_workers)
        self.cache_path = cache_path or path.join(model_path, ".cache")

    @classmethod
    def from_environ(cls):
        num_workers = int(environ.get("ONNX_WEB_NUM_WORKERS", 1))
        cache_limit = int(environ.get("ONNX_WEB_CACHE_MODELS", num_workers + 2))

        return ServerContext(
            bundle_path=environ.get(
                "ONNX_WEB_BUNDLE_PATH", path.join("..", "gui", "out")
            ),
            model_path=environ.get("ONNX_WEB_MODEL_PATH", path.join("..", "models")),
            output_path=environ.get("ONNX_WEB_OUTPUT_PATH", path.join("..", "outputs")),
            params_path=environ.get("ONNX_WEB_PARAMS_PATH", "."),
            # others
            cors_origin=environ.get("ONNX_WEB_CORS_ORIGIN", "*").split(","),
            num_workers=num_workers,
            any_platform=get_boolean(environ, "ONNX_WEB_ANY_PLATFORM", True),
            block_platforms=environ.get("ONNX_WEB_BLOCK_PLATFORMS", "").split(","),
            default_platform=environ.get("ONNX_WEB_DEFAULT_PLATFORM", None),
            image_format=environ.get("ONNX_WEB_IMAGE_FORMAT", "png"),
            cache=ModelCache(limit=cache_limit),
        )


def base_join(base: str, tail: str) -> str:
    tail_path = path.relpath(path.normpath(path.join("/", tail)), "/")
    return path.join(base, tail_path)


def is_debug() -> bool:
    return get_boolean(environ, "DEBUG", False)


def get_boolean(args: Any, key: str, default_value: bool) -> bool:
    return args.get(key, str(default_value)).lower() in ("1", "t", "true", "y", "yes")


def get_and_clamp_float(
    args: Any, key: str, default_value: float, max_value: float, min_value=0.0
) -> float:
    return min(max(float(args.get(key, default_value)), min_value), max_value)


def get_and_clamp_int(
    args: Any, key: str, default_value: int, max_value: int, min_value=1
) -> int:
    return min(max(int(args.get(key, default_value)), min_value), max_value)


def get_from_list(args: Any, key: str, values: List[Any]) -> Optional[Any]:
    selected = args.get(key, None)
    if selected in values:
        return selected

    logger.warn("invalid selection: %s", selected)
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

    if type(val) is int:
        return val

    if type(val) is str:
        for size in SizeChart:
            if val == size.name:
                return size

        return int(val)

    raise Exception("invalid size")


def run_gc():
    logger.debug("running garbage collection")
    gc.collect()
    torch.cuda.empty_cache()
