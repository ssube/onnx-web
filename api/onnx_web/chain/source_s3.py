from io import BytesIO
from logging import getLogger
from typing import List, Optional

from boto3 import Session
from PIL import Image

from ..params import ImageParams, StageParams
from ..server import ServerContext
from ..worker import WorkerContext, ProgressCallback
from .base import BaseStage
from .result import ImageMetadata, StageResult

logger = getLogger(__name__)


class SourceS3Stage(BaseStage):
    def run(
        self,
        _worker: WorkerContext,
        _server: ServerContext,
        _stage: StageParams,
        _params: ImageParams,
        sources: StageResult,
        *,
        source_keys: List[str],
        bucket: str,
        endpoint_url: Optional[str] = None,
        profile_name: Optional[str] = None,
        callback: Optional[ProgressCallback] = None,
        **kwargs,
    ) -> StageResult:
        session = Session(profile_name=profile_name)
        s3 = session.client("s3", endpoint_url=endpoint_url)

        if len(sources) > 0:
            logger.info(
                "source images were passed to a source stage, new images will be appended"
            )

        outputs = sources.as_images()
        for key in source_keys:
            try:
                logger.info("loading image from s3://%s/%s", bucket, key)
                data = BytesIO()
                s3.download_fileobj(bucket, key, data)

                data.seek(0)
                outputs.append(Image.open(data))
            except Exception:
                logger.exception("error loading image from S3")

        # TODO: attempt to load metadata from s3 or load it from the image itself (exif data)
        metadata = [ImageMetadata.unknown_image()] * len(outputs)
        return StageResult(outputs, metadata=metadata)

    def outputs(
        self,
        params: ImageParams,
        sources: int,
    ) -> int:
        return sources + 1  # TODO: len(source_keys)
