from ..utils import (
    ConversionContext,
    build_cache_paths,
    download_progress,
    get_first_exists,
    remove_prefix,
)
from .base import BaseClient
from typing import Dict, Optional
from logging import getLogger

logger = getLogger(__name__)


class HttpClient(BaseClient):
    name = "http"
    protocol = "https://"
    insecure_protocol = "http://"

    headers: Dict[str, str]

    def __init__(self, headers: Optional[Dict[str, str]] = None):
        self.headers = headers or {}

    def download(self, conversion: ConversionContext, name: str, uri: str) -> str:
        cache_paths = build_cache_paths(
            conversion, name, client=HttpClient.name, format=format
        )
        cached = get_first_exists(cache_paths)
        if cached:
            return cached

        if uri.startswith(HttpClient.protocol):
            source = remove_prefix(uri, HttpClient.protocol)
            logger.info("downloading model from: %s", source)
        elif uri.startswith(HttpClient.insecure_protocol):
            logger.warning("downloading model from insecure source: %s", source)

        return download_progress(source, cache_paths[0])
