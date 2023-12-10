from ..utils import (
    ConversionContext,
    build_cache_paths,
    get_first_exists,
    remove_prefix,
)
from .base import BaseClient
from typing import Optional, Any
from logging import getLogger
from huggingface_hub.file_download import hf_hub_download

logger = getLogger(__name__)


class HuggingfaceClient(BaseClient):
    name = "huggingface"
    protocol = "huggingface://"

    download: Any
    token: Optional[str]

    def __init__(self, token: Optional[str] = None, download=hf_hub_download):
        self.download = download
        self.token = token

    def download(
        self,
        conversion: ConversionContext,
        name: str,
        source: str,
        format: Optional[str],
    ) -> str:
        """
        TODO: download with auth
        """
        hf_hub_fetch = TODO
        hf_hub_filename = TODO

        cache_paths = build_cache_paths(
            conversion, name, client=HuggingfaceClient.name, format=format
        )
        cached = get_first_exists(cache_paths)
        if cached:
            return cached

        source = remove_prefix(source, HuggingfaceClient.protocol)
        logger.info("downloading model from Huggingface Hub: %s", source)

        # from_pretrained has a bunch of useful logic that snapshot_download by itself down not
        if hf_hub_fetch:
            return (
                hf_hub_download(
                    repo_id=source,
                    filename=hf_hub_filename,
                    cache_dir=cache_paths[0],
                    force_filename=f"{name}.bin",
                ),
                False,
            )
        else:
            return source
