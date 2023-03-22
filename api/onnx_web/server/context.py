from logging import getLogger
from os import environ, path
from typing import List, Optional

import torch

from ..utils import get_boolean
from .model_cache import ModelCache

logger = getLogger(__name__)

DEFAULT_CACHE_LIMIT = 5
DEFAULT_JOB_LIMIT = 10
DEFAULT_IMAGE_FORMAT = "png"


class ServerContext:
    def __init__(
        self,
        bundle_path: str = ".",
        model_path: str = ".",
        output_path: str = ".",
        params_path: str = ".",
        cors_origin: str = "*",
        any_platform: bool = True,
        block_platforms: Optional[List[str]] = None,
        default_platform: Optional[str] = None,
        image_format: str = DEFAULT_IMAGE_FORMAT,
        cache_limit: int = DEFAULT_CACHE_LIMIT,
        cache_path: Optional[str] = None,
        show_progress: bool = True,
        optimizations: Optional[List[str]] = None,
        extra_models: Optional[List[str]] = None,
        job_limit: int = DEFAULT_JOB_LIMIT,
        memory_limit: Optional[int] = None,
    ) -> None:
        self.bundle_path = bundle_path
        self.model_path = model_path
        self.output_path = output_path
        self.params_path = params_path
        self.cors_origin = cors_origin
        self.any_platform = any_platform
        self.block_platforms = block_platforms or []
        self.default_platform = default_platform
        self.image_format = image_format
        self.cache_limit = cache_limit
        self.cache_path = cache_path or path.join(model_path, ".cache")
        self.show_progress = show_progress
        self.optimizations = optimizations or []
        self.extra_models = extra_models or []
        self.job_limit = job_limit
        self.memory_limit = memory_limit

        self.cache = ModelCache(self.cache_limit)

    @classmethod
    def from_environ(cls):
        memory_limit = environ.get("ONNX_WEB_MEMORY_LIMIT", None)
        if memory_limit is not None:
            memory_limit = int(memory_limit)

        return cls(
            bundle_path=environ.get(
                "ONNX_WEB_BUNDLE_PATH", path.join("..", "gui", "out")
            ),
            model_path=environ.get("ONNX_WEB_MODEL_PATH", path.join("..", "models")),
            output_path=environ.get("ONNX_WEB_OUTPUT_PATH", path.join("..", "outputs")),
            params_path=environ.get("ONNX_WEB_PARAMS_PATH", "."),
            # others
            cors_origin=environ.get("ONNX_WEB_CORS_ORIGIN", "*").split(","),
            any_platform=get_boolean(environ, "ONNX_WEB_ANY_PLATFORM", True),
            block_platforms=environ.get("ONNX_WEB_BLOCK_PLATFORMS", "").split(","),
            default_platform=environ.get("ONNX_WEB_DEFAULT_PLATFORM", None),
            image_format=environ.get("ONNX_WEB_IMAGE_FORMAT", "png"),
            cache_limit=int(environ.get("ONNX_WEB_CACHE_MODELS", DEFAULT_CACHE_LIMIT)),
            show_progress=get_boolean(environ, "ONNX_WEB_SHOW_PROGRESS", True),
            optimizations=environ.get("ONNX_WEB_OPTIMIZATIONS", "").split(","),
            extra_models=environ.get("ONNX_WEB_EXTRA_MODELS", "").split(","),
            job_limit=int(environ.get("ONNX_WEB_JOB_LIMIT", DEFAULT_JOB_LIMIT)),
            memory_limit=memory_limit,
        )

    def torch_dtype(self):
        if "torch-fp16" in self.optimizations:
            return torch.float16
        else:
            return torch.float32
