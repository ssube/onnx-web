from logging import getLogger
from os import path
from typing import Optional

import cv2
import torch
from PIL import Image
from torchvision.transforms.functional import normalize

from ..params import ImageParams, StageParams, UpscaleParams
from ..server import ServerContext
from ..worker import WorkerContext
from .base import BaseStage
from .result import StageResult

logger = getLogger(__name__)

CORRECTION_MODEL = "correction-codeformer.pth"
DETECTION_MODEL = "retinaface_resnet50"


class CorrectCodeformerStage(BaseStage):
    def run(
        self,
        worker: WorkerContext,
        server: ServerContext,
        _stage: StageParams,
        _params: ImageParams,
        sources: StageResult,
        *,
        stage_source: Optional[Image.Image] = None,
        upscale: UpscaleParams,
        **kwargs,
    ) -> StageResult:
        # adapted from https://github.com/kadirnar/codeformer-pip/blob/main/codeformer/app.py and
        # https://pypi.org/project/codeformer-perceptor/

        # import must be within the load function for patches to take effect
        # TODO: rewrite and remove
        from codeformer.basicsr.utils import img2tensor, tensor2img
        from codeformer.basicsr.utils.registry import ARCH_REGISTRY
        from codeformer.facelib.utils.face_restoration_helper import FaceRestoreHelper

        upscale = upscale.with_args(**kwargs)
        device = worker.get_device()

        net = ARCH_REGISTRY.get("CodeFormer")(
            dim_embd=512,
            codebook_size=1024,
            n_head=8,
            n_layers=9,
            connect_list=["32", "64", "128", "256"],
        ).to(device.torch_str())

        ckpt_path = path.join(server.cache_path, CORRECTION_MODEL)
        checkpoint = torch.load(ckpt_path)["params_ema"]
        net.load_state_dict(checkpoint)
        net.eval()

        face_helper = FaceRestoreHelper(
            upscale.face_outscale,
            face_size=512,
            crop_ratio=(1, 1),
            det_model=DETECTION_MODEL,
            save_ext="png",
            use_parse=True,
            device=device.torch_str(),
        )

        results = []
        for img in sources.as_numpy():
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            # clean all the intermediate results to process the next image
            face_helper.clean_all()
            face_helper.read_image(img)

            # get face landmarks for each face
            num_det_faces = face_helper.get_face_landmarks_5(
                only_center_face=False, resize=640, eye_dist_threshold=5
            )
            logger.debug("detected %s faces", num_det_faces)

            # align and warp each face
            face_helper.align_warp_face()

            # face restoration for each cropped face
            for cropped_face in face_helper.cropped_faces:
                # prepare data
                cropped_face_t = img2tensor(
                    cropped_face / 255.0, bgr2rgb=True, float32=True
                )
                normalize(
                    cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True
                )
                cropped_face_t = cropped_face_t.unsqueeze(0).to(device.torch_str())

                try:
                    with torch.no_grad():
                        output = net(
                            cropped_face_t, w=upscale.face_strength, adain=True
                        )[0]
                        restored_face = tensor2img(
                            output, rgb2bgr=True, min_max=(-1, 1)
                        )

                    del output
                except Exception:
                    logger.exception("inference failed for CodeFormer")
                    restored_face = tensor2img(
                        cropped_face_t, rgb2bgr=True, min_max=(-1, 1)
                    )

                restored_face = restored_face.astype("uint8")
                face_helper.add_restored_face(restored_face, cropped_face)

            # paste_back
            face_helper.get_inverse_affine(None)

            # paste each restored face to the input image
            output = face_helper.paste_faces_to_input_image(upsample_img=img, draw_box=False)
            results.append(Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB)))

        return StageResult.from_images(results)
