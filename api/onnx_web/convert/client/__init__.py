from .base import BaseClient
from .civitai import CivitaiClient
from .file import FileClient
from .http import HttpClient
from .huggingface import HuggingfaceClient
from ..utils import ConversionContext
from typing import Dict, Optional
from logging import getLogger
from os import path

logger = getLogger(__name__)


model_sources: Dict[str, BaseClient] = {
    CivitaiClient.protocol: CivitaiClient,
    FileClient.protocol: FileClient,
    HttpClient.insecure_protocol: HttpClient,
    HttpClient.protocol: HttpClient,
    HuggingfaceClient.protocol: HuggingfaceClient,
}


def add_model_source(proto: str, client: BaseClient):
    global model_sources

    if proto in model_sources:
        raise ValueError("protocol has already been taken")

    model_sources[proto] = client


def fetch_model(
    conversion: ConversionContext,
    name: str,
    source: str,
    format: Optional[str] = None,
    dest: Optional[str] = None,
    **kwargs,
) -> str:
    # TODO: switch to urlparse's default scheme
    if source.startswith(path.sep) or source.startswith("."):
        logger.info("adding file protocol to local path source: %s", source)
        source = FileClient.protocol + source

    for proto, client_type in model_sources.items():
        if source.startswith(proto):
            # TODO: fix type of client_type
            client: BaseClient = client_type()
            return client.download(conversion, name, source, format=format, dest=dest, **kwargs)

    logger.warning("unknown model protocol, using path as provided: %s", source)
    return source
