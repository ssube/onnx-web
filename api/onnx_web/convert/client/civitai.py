from logging import getLogger
from typing import Optional

from ..utils import (
    ConversionContext,
    build_cache_paths,
    download_progress,
    get_first_exists,
    remove_prefix,
)
from .base import BaseClient

logger = getLogger(__name__)

CIVITAI_ROOT = "https://civitai.com/api/download/models/%s"


class CivitaiClient(BaseClient):
    name = "civitai"
    protocol = "civitai://"

    root: str
    token: Optional[str]

    def __init__(
        self,
        conversion: ConversionContext,
        token: Optional[str] = None,
        root: str = CIVITAI_ROOT,
    ):
        self.root = conversion.get_setting("CIVITAI_ROOT", root)
        self.token = conversion.get_setting("CIVITAI_TOKEN", token)

    def download(
        self,
        conversion: ConversionContext,
        name: str,
        source: str,
        format: Optional[str] = None,
        dest: Optional[str] = None,
        **kwargs,
    ) -> str:
        cache_paths = build_cache_paths(
            conversion,
            name,
            client=CivitaiClient.name,
            format=format,
            dest=dest,
        )
        cached = get_first_exists(cache_paths)
        if cached:
            return cached

        source = self.root % (remove_prefix(source, CivitaiClient.protocol))
        logger.info("downloading model from Civitai: %s -> %s", source, cache_paths[0])

        if self.token:
            logger.debug("adding Civitai token authentication")
            if "?" in source:
                source = f"{source}&token={self.token}"
            else:
                source = f"{source}?token={self.token}"

        return download_progress(source, cache_paths[0])
