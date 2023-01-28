from boto3 import (
    Session,
)
from io import BytesIO
from PIL import Image

from ..params import (
    ImageParams,
    StageParams,
)
from ..utils import (
    ServerContext,
)


def persist_s3(
    _ctx: ServerContext,
    _stage: StageParams,
    _params: ImageParams,
    source_image: Image.Image,
    *,
    output: str,
    bucket: str,
    endpoint_url: str = None,
    profile_name: str = None,
) -> Image.Image:
    session = Session(profile_name=profile_name)
    s3 = session.client('s3', endpoint_url=endpoint_url)

    data = BytesIO()
    source_image.save(data, format='png')
    data.seek(0)

    try:
        response = s3.upload_fileobj(data, bucket, output)
        print('saved image to %s' % (response))
    except Exception as err:
        print('error saving image to S3: %s' % (err))

    return source_image
