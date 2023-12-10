from logging import getLogger
from typing import Any, Optional

from huggingface_hub.file_download import hf_hub_download

from ..utils import (
    ConversionContext,
    build_cache_paths,
    get_first_exists,
    remove_prefix,
)
from .base import BaseClient

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
        format: Optional[str] = None,
        dest: Optional[str] = None,
    ) -> str:
        """
        TODO: download with auth
        TODO: set fetch and filename
            if network_type == "inversion" and network_model == "concept":
        """
        hf_hub_fetch = True
        hf_hub_filename = "learned_embeds.bin"

        cache_paths = build_cache_paths(
            conversion,
            name,
            client=HuggingfaceClient.name,
            format=format,
            dest=dest,
        )
        cached = get_first_exists(cache_paths)
        if cached:
            return cached

        source = remove_prefix(source, HuggingfaceClient.protocol)
        logger.info("downloading model from Huggingface Hub: %s", source)

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
            # TODO: download pretrained because load doesn't call from_pretrained anymore
            return source
