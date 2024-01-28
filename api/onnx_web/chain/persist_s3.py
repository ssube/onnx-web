from io import BytesIO
from json import dumps
from logging import getLogger
from typing import Optional

from boto3 import Session
from PIL import Image

from ..output import make_output_names
from ..params import ImageParams, StageParams
from ..server import ServerContext
from ..worker import ProgressCallback, WorkerContext
from .base import BaseStage
from .result import StageResult

logger = getLogger(__name__)


class PersistS3Stage(BaseStage):
    def run(
        self,
        worker: WorkerContext,
        server: ServerContext,
        _stage: StageParams,
        _params: ImageParams,
        sources: StageResult,
        *,
        bucket: str,
        endpoint_url: Optional[str] = None,
        profile_name: Optional[str] = None,
        stage_source: Optional[Image.Image] = None,
        callback: Optional[ProgressCallback] = None,
        **kwargs,
    ) -> StageResult:
        session = Session(profile_name=profile_name)
        s3 = session.client("s3", endpoint_url=endpoint_url)

        image_names = make_output_names(server, worker.job, len(sources))
        for source, name in zip(sources.as_images(), image_names):
            data = BytesIO()
            source.save(data, format=server.image_format)
            data.seek(0)

            try:
                s3.upload_fileobj(data, bucket, name)
                logger.info("saved image to s3://%s/%s", bucket, name)
            except Exception:
                logger.exception("error saving image to S3")

        metadata_names = make_output_names(
            server, worker.job, len(sources), extension="json"
        )
        for metadata, name in zip(sources.metadata, metadata_names):
            data = BytesIO()
            data.write(dumps(metadata.tojson(server, [name])))
            data.seek(0)

            try:
                s3.upload_fileobj(data, bucket, name)
                logger.info("saved metadata to s3://%s/%s", bucket, name)
            except Exception:
                logger.exception("error saving metadata to S3")

        return sources
