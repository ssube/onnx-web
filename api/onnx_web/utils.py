import gc
from logging import getLogger
from os import environ, path
from typing import Any, Dict, List, Optional, Union

import torch

from .params import SizeChart

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
        block_platforms: List[str] = [],
        default_platform: str = None,
        image_format: str = "png",
    ) -> None:
        self.bundle_path = bundle_path
        self.model_path = model_path
        self.output_path = output_path
        self.params_path = params_path
        self.cors_origin = cors_origin
        self.num_workers = num_workers
        self.block_platforms = block_platforms
        self.default_platform = default_platform
        self.image_format = image_format

    @classmethod
    def from_environ(cls):
        return ServerContext(
            bundle_path=environ.get(
                "ONNX_WEB_BUNDLE_PATH", path.join("..", "gui", "out")
            ),
            model_path=environ.get("ONNX_WEB_MODEL_PATH", path.join("..", "models")),
            output_path=environ.get("ONNX_WEB_OUTPUT_PATH", path.join("..", "outputs")),
            params_path=environ.get("ONNX_WEB_PARAMS_PATH", "."),
            # others
            cors_origin=environ.get("ONNX_WEB_CORS_ORIGIN", "*").split(","),
            num_workers=int(environ.get("ONNX_WEB_NUM_WORKERS", 1)),
            block_platforms=environ.get("ONNX_WEB_BLOCK_PLATFORMS", "").split(","),
            default_platform=environ.get("ONNX_WEB_DEFAULT_PLATFORM", None),
            image_format=environ.get("ONNX_WEB_IMAGE_FORMAT", "png"),
        )


def base_join(base: str, tail: str) -> str:
    tail_path = path.relpath(path.normpath(path.join("/", tail)), "/")
    return path.join(base, tail_path)


def is_debug() -> bool:
    return environ.get("DEBUG") is not None


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
