from boto3 import (
    ClientError,
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
    sess = Session(profile_name=profile_name)
    s3 = sess.client('s3', endpoint_url=endpoint_url)

    data = BytesIO()
    source_image.save(data, format='png')

    try:
        response = s3.upload_fileobj(data.getvalue(), bucket, output)
        print('saved image to %s' % (response))
    except ClientError as err:
        print('error saving image to S3: %s' % (err))

    return source_image
