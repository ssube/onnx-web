# https://github.com/ForserX/StableDiffusionUI/blob/main/data/repo/diffusion_scripts/controlnet_pipe.py

from logging import getLogger
from os import path

import cv2
import numpy as np
import torch
import transformers
from controlnet_aux import HEDdetector, MLSDdetector, OpenposeDetector
from huggingface_hub import snapshot_download
from PIL import Image

from ..server.context import ServerContext
from .laion_face import generate_annotation
from .utils import ade_palette

logger = getLogger(__name__)


def pil_to_cv2(source: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(source), cv2.COLOR_RGB2BGR)


def filter_model_path(server: ServerContext, filter_name: str) -> str:
    return path.join(server.model_path, "filter", filter_name)


def source_filter_gaussian():
    pass


def source_filter_noise():
    pass


def source_filter_face(
    server: ServerContext,
    source: Image.Image,
    max_faces: int = 1,
    min_confidence: float = 0.5,
) -> Image.Image:
    logger.debug("running face detection on source image")

    image = generate_annotation(pil_to_cv2(source), max_faces, min_confidence)
    image = Image.fromarray(image)

    return image


def source_filter_segment(server: ServerContext, source: Image.Image) -> Image.Image:
    logger.debug("running segmentation on source image")

    openmm_model = snapshot_download(
        "openmmlab/upernet-convnext-small",
        allow_patterns=["*.bin", "*.json"],
        cache_dir=filter_model_path(server, "upernet-convnext-small"),
    )

    image_processor = transformers.AutoImageProcessor.from_pretrained(openmm_model)
    image_segmentor = transformers.UperNetForSemanticSegmentation.from_pretrained(
        openmm_model
    )

    in_img = source.convert("RGB")

    pixel_values = image_processor(in_img, return_tensors="pt").pixel_values

    with torch.no_grad():
        outputs = image_segmentor(pixel_values)

    seg = image_processor.post_process_semantic_segmentation(
        outputs, target_sizes=[in_img.size[::-1]]
    )[0]

    color_seg = np.zeros(
        (seg.shape[0], seg.shape[1], 3), dtype=np.uint8
    )  # height, width, 3

    palette = np.array(ade_palette())

    for label, color in enumerate(palette):
        color_seg[seg == label, :] = color

    color_seg = color_seg.astype(np.uint8)

    image = Image.fromarray(color_seg)

    return image


def source_filter_mlsd(server: ServerContext, source: Image.Image) -> Image.Image:
    logger.debug("running MLSD on source image")

    mlsd = MLSDdetector.from_pretrained(
        "lllyasviel/ControlNet",
        cache_dir=server.cache_path,
    )
    image = mlsd(source)

    return image


def source_filter_normal(server: ServerContext, source: Image.Image) -> Image.Image:
    logger.debug("running normal detection on source image")

    depth_estimator = transformers.pipeline(
        "depth-estimation",
        model=snapshot_download(
            "Intel/dpt-hybrid-midas",
            allow_patterns=["*.bin", "*.json"],
            cache_dir=filter_model_path(server, "dpt-hybrid-midas"),
        ),
    )

    image = depth_estimator(source)["predicted_depth"][0]

    image = image.numpy()

    image_depth = image.copy()
    image_depth -= np.min(image_depth)
    image_depth /= np.max(image_depth)

    bg_threhold = 0.4

    x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    x[image_depth < bg_threhold] = 0

    y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    y[image_depth < bg_threhold] = 0

    z = np.ones_like(x) * np.pi * 2.0

    image = np.stack([x, y, z], axis=2)
    image /= np.sum(image**2.0, axis=2, keepdims=True) ** 0.5
    image = (image * 127.5 + 127.5).clip(0, 255).astype(np.uint8)
    image = Image.fromarray(image)

    return image


def source_filter_hed(server: ServerContext, source: Image.Image) -> Image.Image:
    logger.debug("running HED detection on source image")

    hed = HEDdetector.from_pretrained(
        "lllyasviel/ControlNet",
        cache_dir=server.cache_path,
    )
    image = hed(source)

    return image


def source_filter_scribble(server: ServerContext, source: Image.Image) -> Image.Image:
    logger.debug("running scribble detection on source image")

    hed = HEDdetector.from_pretrained(
        "lllyasviel/ControlNet",
        cache_dir=server.cache_path,
    )
    image = hed(source, scribble=True)

    return image


def source_filter_depth(server: ServerContext, source: Image.Image) -> Image.Image:
    logger.debug("running depth detection on source image")
    depth_estimator = transformers.pipeline("depth-estimation")

    image = depth_estimator(source)["depth"]
    image = np.array(image)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    image = Image.fromarray(image)

    return image


def source_filter_canny(
    server: ServerContext, source: Image.Image, low_threshold=100, high_threshold=200
) -> Image.Image:
    logger.debug("running Canny detection on source image")

    image = cv2.Canny(pil_to_cv2(source), low_threshold, high_threshold)
    image = Image.fromarray(image)

    return image


def source_filter_openpose(server: ServerContext, source: Image.Image) -> Image.Image:
    logger.debug("running OpenPose detection on source image")

    model = OpenposeDetector.from_pretrained(
        "lllyasviel/ControlNet",
        cache_dir=server.cache_dir,
    )
    image = model(source)

    return image
