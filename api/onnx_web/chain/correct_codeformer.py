from logging import getLogger

import torch
from basicsr.utils import img2tensor, tensor2img
from basicsr.utils.download_util import load_file_from_url
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
from PIL import Image
from torchvision.transforms.functional import normalize

from ..device_pool import JobContext
from ..params import ImageParams, StageParams, UpscaleParams
from ..utils import ServerContext

logger = getLogger(__name__)

pretrain_model_url = (
    "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth"
)

device = "cpu"
upscale = 2


def correct_codeformer(
    job: JobContext,
    server: ServerContext,
    stage: StageParams,
    params: ImageParams,
    source_image: Image.Image,
    *,
    upscale: UpscaleParams = None,
    **kwargs,
) -> Image.Image:
    ARCH_REGISTRY = {}
    bg_upsampler = None
    face_upsampler = None
    model = "TODO"
    w = None

    # ------------------ set up CodeFormer restorer -------------------
    net = ARCH_REGISTRY.get("CodeFormer")(
        dim_embd=512,
        codebook_size=1024,
        n_head=8,
        n_layers=9,
        connect_list=["32", "64", "128", "256"],
    ).to(device)

    # ckpt_path = 'weights/CodeFormer/codeformer.pth'
    ckpt_path = load_file_from_url(
        url=pretrain_model_url,
        model_dir="weights/CodeFormer",
        progress=True,
        file_name=None,
    )
    checkpoint = torch.load(ckpt_path)
    checkpoint = checkpoint["params_ema"]
    net.load_state_dict(checkpoint)
    net.eval()

    # ------------------ set up FaceRestoreHelper -------------------
    # large det_model: 'YOLOv5l', 'retinaface_resnet50'
    # small det_model: 'YOLOv5n', 'retinaface_mobile0.25'

    face_helper = FaceRestoreHelper(
        upscale,
        face_size=512,
        crop_ratio=(1, 1),
        det_model=model,
        save_ext="png",
        use_parse=True,
        device=device,
    )

    # get face landmarks for each face
    num_det_faces = face_helper.get_face_landmarks_5(
        only_center_face=False, resize=640, eye_dist_threshold=5
    )
    logger.info("detect %s faces", num_det_faces)
    # align and warp each face
    face_helper.align_warp_face()

    # face restoration for each cropped face
    for idx, cropped_face in enumerate(face_helper.cropped_faces):
        # prepare data
        cropped_face_t = img2tensor(cropped_face / 255.0, bgr2rgb=True, float32=True)
        normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        cropped_face_t = cropped_face_t.unsqueeze(0).to(device)

        try:
            with torch.no_grad():
                output = net(cropped_face_t, w=w, adain=True)[0]
                restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
            del output
            torch.cuda.empty_cache()
        except Exception as error:
            logger.error("Failed inference for CodeFormer: %s", error)
            restored_face = tensor2img(cropped_face_t, rgb2bgr=True, min_max=(-1, 1))

        restored_face = restored_face.astype("uint8")
        face_helper.add_restored_face(restored_face, cropped_face)

    # upsample the background
    if bg_upsampler is not None:
        # Now only support RealESRGAN for upsampling background
        bg_img = bg_upsampler.enhance(source_image, outscale=upscale.scale)[0]
    else:
        bg_img = None

    # paste_back
    face_helper.get_inverse_affine(None)
    # paste each restored face to the input image
    if face_upsampler is not None:
        restored_img = face_helper.paste_faces_to_input_image(
            upsample_img=bg_img, draw_box=False, face_upsampler=face_upsampler
        )
    else:
        restored_img = face_helper.paste_faces_to_input_image(
            upsample_img=bg_img, draw_box=False
        )

    return restored_img
