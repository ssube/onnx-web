from logging import getLogger
from os import environ, path
from secrets import token_urlsafe
from typing import List, Optional

import torch

from ..utils import get_boolean
from .model_cache import ModelCache

logger = getLogger(__name__)

DEFAULT_ANY_PLATFORM = True
DEFAULT_CACHE_LIMIT = 5
DEFAULT_JOB_LIMIT = 10
DEFAULT_IMAGE_FORMAT = "png"
DEFAULT_SERVER_VERSION = "v0.10.0"
DEFAULT_SHOW_PROGRESS = True
DEFAULT_PANORAMA_TILES = False
DEFAULT_WORKER_RETRIES = 3


class ServerContext:
    bundle_path: str
    model_path: str
    output_path: str
    params_path: str
    cors_origin: str
    any_platform: bool
    block_platforms: List[str]
    default_platform: str
    image_format: str
    cache_limit: int
    cache_path: str
    show_progress: bool
    optimizations: List[str]
    extra_models: List[str]
    job_limit: int
    memory_limit: int
    admin_token: str
    server_version: str
    worker_retries: int
    panorama_tiles: bool

    def __init__(
        self,
        bundle_path: str = ".",
        model_path: str = ".",
        output_path: str = ".",
        params_path: str = ".",
        cors_origin: str = "*",
        any_platform: bool = DEFAULT_ANY_PLATFORM,
        block_platforms: Optional[List[str]] = None,
        default_platform: Optional[str] = None,
        image_format: str = DEFAULT_IMAGE_FORMAT,
        cache_limit: int = DEFAULT_CACHE_LIMIT,
        cache_path: Optional[str] = None,
        show_progress: bool = DEFAULT_SHOW_PROGRESS,
        optimizations: Optional[List[str]] = None,
        extra_models: Optional[List[str]] = None,
        job_limit: int = DEFAULT_JOB_LIMIT,
        memory_limit: Optional[int] = None,
        admin_token: Optional[str] = None,
        server_version: Optional[str] = DEFAULT_SERVER_VERSION,
        worker_retries: Optional[int] = DEFAULT_WORKER_RETRIES,
        panorama_tiles: Optional[bool] = DEFAULT_PANORAMA_TILES,
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
        self.admin_token = admin_token or token_urlsafe()
        self.server_version = server_version
        self.worker_retries = worker_retries
        self.panorama_tiles = panorama_tiles

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
            any_platform=get_boolean(
                environ, "ONNX_WEB_ANY_PLATFORM", DEFAULT_ANY_PLATFORM
            ),
            block_platforms=environ.get("ONNX_WEB_BLOCK_PLATFORMS", "").split(","),
            default_platform=environ.get("ONNX_WEB_DEFAULT_PLATFORM", None),
            image_format=environ.get("ONNX_WEB_IMAGE_FORMAT", "png"),
            cache_limit=int(environ.get("ONNX_WEB_CACHE_MODELS", DEFAULT_CACHE_LIMIT)),
            show_progress=get_boolean(
                environ, "ONNX_WEB_SHOW_PROGRESS", DEFAULT_SHOW_PROGRESS
            ),
            optimizations=environ.get("ONNX_WEB_OPTIMIZATIONS", "").split(","),
            extra_models=environ.get("ONNX_WEB_EXTRA_MODELS", "").split(","),
            job_limit=int(environ.get("ONNX_WEB_JOB_LIMIT", DEFAULT_JOB_LIMIT)),
            memory_limit=memory_limit,
            admin_token=environ.get("ONNX_WEB_ADMIN_TOKEN", None),
            server_version=environ.get(
                "ONNX_WEB_SERVER_VERSION", DEFAULT_SERVER_VERSION
            ),
            worker_retries=int(
                environ.get("ONNX_WEB_WORKER_RETRIES", DEFAULT_WORKER_RETRIES)
            ),
            panorama_tiles=get_boolean(
                environ, "ONNX_WEB_PANORAMA_TILES", DEFAULT_PANORAMA_TILES
            ),
        )

    def torch_dtype(self):
        if "torch-fp16" in self.optimizations:
            return torch.float16
        else:
            return torch.float32
