from io import BytesIO
from logging import getLogger
from typing import Optional

from boto3 import Session
from PIL import Image

from ..params import ImageParams, StageParams
from ..server import ServerContext
from ..worker import WorkerContext

logger = getLogger(__name__)


def persist_s3(
    _job: WorkerContext,
    server: ServerContext,
    _stage: StageParams,
    _params: ImageParams,
    source: Image.Image,
    *,
    output: str,
    bucket: str,
    endpoint_url: Optional[str] = None,
    profile_name: Optional[str] = None,
    stage_source: Optional[Image.Image] = None,
    **kwargs,
) -> Image.Image:
    source = stage_source or source

    session = Session(profile_name=profile_name)
    s3 = session.client("s3", endpoint_url=endpoint_url)

    data = BytesIO()
    source.save(data, format=server.image_format)
    data.seek(0)

    try:
        s3.upload_fileobj(data, bucket, output)
        logger.info("saved image to %s/%s", bucket, output)
    except Exception as err:
        logger.error("error saving image to S3: %s", err)

    return source
