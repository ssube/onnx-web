from logging import getLogger
from os import path
from typing import Optional
from urllib.parse import urlparse

from ..utils import ConversionContext
from .base import BaseClient

logger = getLogger(__name__)


class FileClient(BaseClient):
    protocol = "file://"

    root: str

    def __init__(self, root: str):
        self.root = root

    def download(
        self,
        _conversion: ConversionContext,
        _name: str,
        uri: str,
        format: Optional[str] = None,
        dest: Optional[str] = None,
        **kwargs,
    ) -> str:
        parts = urlparse(uri)
        logger.info("loading model from: %s", parts.path)
        return path.join(self.root, parts.path)
