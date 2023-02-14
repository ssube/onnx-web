from io import BytesIO
from logging import getLogger

from boto3 import Session
from PIL import Image

from ..params import ImageParams, StageParams
from ..server.device_pool import JobContext
from ..utils import ServerContext

logger = getLogger(__name__)


def persist_s3(
    _job: JobContext,
    server: ServerContext,
    _stage: StageParams,
    _params: ImageParams,
    source_image: Image.Image,
    *,
    output: str,
    bucket: str,
    endpoint_url: str = None,
    profile_name: str = None,
    **kwargs,
) -> Image.Image:
    session = Session(profile_name=profile_name)
    s3 = session.client("s3", endpoint_url=endpoint_url)

    data = BytesIO()
    source_image.save(data, format=server.image_format)
    data.seek(0)

    try:
        s3.upload_fileobj(data, bucket, output)
        logger.info("saved image to %s/%s", bucket, output)
    except Exception as err:
        logger.error("error saving image to S3: %s", err)

    return source_image
