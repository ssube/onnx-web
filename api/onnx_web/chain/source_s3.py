from io import BytesIO
from logging import getLogger
from typing import Optional

from boto3 import Session
from PIL import Image

from ..params import ImageParams, StageParams
from ..server import ServerContext
from ..worker import WorkerContext

logger = getLogger(__name__)


def source_s3(
    _job: WorkerContext,
    server: ServerContext,
    _stage: StageParams,
    _params: ImageParams,
    source: Image.Image,
    *,
    source_key: str,
    bucket: str,
    endpoint_url: Optional[str] = None,
    profile_name: Optional[str] = None,
    stage_source: Optional[Image.Image] = None,
    **kwargs,
) -> Image.Image:
    source = stage_source or source

    session = Session(profile_name=profile_name)
    s3 = session.client("s3", endpoint_url=endpoint_url)

    try:
        logger.info("loading image from s3://%s/%s", bucket, source_key)
        data = BytesIO()
        s3.download_fileobj(bucket, source_key, data)

        data.seek(0)
        return Image.open(data)
    except Exception:
        logger.exception("error loading image from S3")
