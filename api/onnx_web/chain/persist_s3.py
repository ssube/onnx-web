from boto3 import (
    Session,
)
from io import BytesIO
from logging import getLogger
from PIL import Image

from ..device_pool import (
    JobContext,
)
from ..params import (
    ImageParams,
    StageParams,
)
from ..utils import (
    ServerContext,
)

logger = getLogger(__name__)


def persist_s3(
    ctx: ServerContext,
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
    s3 = session.client('s3', endpoint_url=endpoint_url)

    data = BytesIO()
    source_image.save(data, format=ctx.image_format)
    data.seek(0)

    try:
        s3.upload_fileobj(data, bucket, output)
        logger.info('saved image to %s/%s', bucket, output)
    except Exception as err:
        logger.error('error saving image to S3: %s', err)

    return source_image
