from io import BytesIO
from logging import getLogger
from typing import List, Optional

from boto3 import Session
from PIL import Image

from ..params import ImageParams, StageParams
from ..server import ServerContext
from ..worker import WorkerContext
from .stage import BaseStage

logger = getLogger(__name__)


class SourceS3Stage(BaseStage):
    def run(
        self,
        _worker: WorkerContext,
        _server: ServerContext,
        _stage: StageParams,
        _params: ImageParams,
        sources: List[Image.Image],
        *,
        source_keys: List[str],
        bucket: str,
        endpoint_url: Optional[str] = None,
        profile_name: Optional[str] = None,
        **kwargs,
    ) -> List[Image.Image]:
        session = Session(profile_name=profile_name)
        s3 = session.client("s3", endpoint_url=endpoint_url)

        if len(sources) > 0:
            logger.info(
                "source images were passed to a source stage, new images will be appended"
            )

        outputs = list(sources)
        for key in source_keys:
            try:
                logger.info("loading image from s3://%s/%s", bucket, key)
                data = BytesIO()
                s3.download_fileobj(bucket, key, data)

                data.seek(0)
                outputs.append(Image.open(data))
            except Exception:
                logger.exception("error loading image from S3")

        return outputs

    def outputs(
            self,
            params: ImageParams,
            sources: int,
    ) -> int:
        return sources + 1 # TODO: len(source_keys)