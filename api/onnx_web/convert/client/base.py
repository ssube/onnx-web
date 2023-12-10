from typing import Optional

from ..utils import ConversionContext


class BaseClient:
    def download(
        self,
        conversion: ConversionContext,
        name: str,
        source: str,
        format: Optional[str] = None,
        dest: Optional[str] = None,
        **kwargs,
    ) -> str:
        raise NotImplementedError()
