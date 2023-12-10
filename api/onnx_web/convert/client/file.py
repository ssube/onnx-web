from .base import BaseClient
from logging import getLogger
from os import path
from urllib.parse import urlparse

logger = getLogger(__name__)


class FileClient(BaseClient):
    protocol = "file://"

    root: str

    def __init__(self, root: str):
        self.root = root

    def download(self, uri: str) -> str:
        parts = urlparse(uri)
        logger.info("loading model from: %s", parts.path)
        return path.join(self.root, parts.path)
