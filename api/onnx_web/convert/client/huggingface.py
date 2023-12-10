from logging import getLogger
from typing import Optional

from huggingface_hub.file_download import hf_hub_download

from ..utils import ConversionContext, remove_prefix
from .base import BaseClient

logger = getLogger(__name__)


class HuggingfaceClient(BaseClient):
    name = "huggingface"
    protocol = "huggingface://"

    token: Optional[str]

    def __init__(self, conversion: ConversionContext, token: Optional[str] = None):
        self.token = conversion.get_setting("HUGGINGFACE_TOKEN", token)

    def download(
        self,
        conversion: ConversionContext,
        name: str,
        source: str,
        format: Optional[str] = None,
        dest: Optional[str] = None,
        embeds: bool = False,
        **kwargs,
    ) -> str:
        source = remove_prefix(source, HuggingfaceClient.protocol)
        logger.info("downloading model from Huggingface Hub: %s", source)

        if embeds:
            return hf_hub_download(
                repo_id=source,
                filename="learned_embeds.bin",
                cache_dir=dest or conversion.cache_path,
                force_filename=f"{name}.bin",
                token=self.token,
            )
        else:
            return source
