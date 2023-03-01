import sys
from functools import partial
from logging import getLogger
from os import path
from urllib.parse import urlparse

import basicsr.utils.download_util
import codeformer.facelib.utils.misc
import facexlib.utils

from .context import ServerContext

logger = getLogger(__name__)


def unload(exclude):
    """
    Remove package modules from cache except excluded ones.
    On next import they will be reloaded.
    From https://medium.com/@chipiga86/python-monkey-patching-like-a-boss-87d7ddb8098e

    Args:
        exclude (iter<str>): Sequence of module paths.
    """
    pkgs = []
    for mod in exclude:
        pkg = mod.split(".", 1)[0]
        pkgs.append(pkg)

    to_unload = []
    for mod in sys.modules:
        if mod in exclude:
            continue

        if mod in pkgs:
            to_unload.append(mod)
            continue

        for pkg in pkgs:
            if mod.startswith(pkg + "."):
                to_unload.append(mod)
                break

    logger.debug("unloading modules for patching: %s", to_unload)
    for mod in to_unload:
        del sys.modules[mod]


# these should be the same sources and names as `convert.base_models.sources`, but inverted so the source is the key
cache_path_map = {
    "https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth": (
        "pt-inception-2015-12-05-6726825d.pth"
    ),
    "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth": (
        "detection-resnet50-final.pth"
    ),
    "https://github.com/xinntao/facexlib/releases/download/v0.1.0/alignment_WFLW_4HG.pth": (
        "alignment-wflw-4hg.pth"
    ),
    "https://github.com/xinntao/facexlib/releases/download/v0.2.0/assessment_hyperIQA.pth": (
        "assessment-hyperiqa.pth"
    ),
    "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_mobilenet0.25_Final.pth": (
        "detection-mobilenet-025-final.pth"
    ),
    "https://github.com/xinntao/facexlib/releases/download/v0.2.0/headpose_hopenet.pth": (
        "headpose-hopenet.pth"
    ),
    "https://github.com/xinntao/facexlib/releases/download/v0.2.0/matting_modnet_portrait.pth": (
        "matting-modnet-portrait.pth"
    ),
    "https://github.com/xinntao/facexlib/releases/download/v0.2.0/parsing_bisenet.pth": (
        "parsing-bisenet.pth"
    ),
    "https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth": (
        "parsing-parsenet.pth"
    ),
    "https://github.com/xinntao/facexlib/releases/download/v0.1.0/recognition_arcface_ir_se50.pth": (
        "recognition-arcface-ir-se50.pth"
    ),
    "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth": (
        "correction-codeformer.pth"
    ),
    "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/detection_Resnet50_Final.pth": (
        "detection-resnet50-final.pth"
    ),
    "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/detection_mobilenet0.25_Final.pth": (
        "detection-mobilenet025-final.pth"
    ),
    "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/parsing_bisenet.pth": (
        "parsing-bisenet.pth"
    ),
    "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/parsing_parsenet.pth": (
        "parsing-parsenet.pth"
    ),
    "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/RealESRGAN_x2plus.pth": (
        "upscaling-real-esrgan-x2-plus"
    ),
    "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/yolov5l-face.pth": (
        "detection-yolo-v5-l.pth"
    ),
    "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/yolov5n-face.pth": (
        "detection-yolo-v5-n.pth"
    ),
    "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth": (
        "correct-gfpgan-v1-3.pth"
    ),
    "https://s3.eu-central-1.wasabisys.com/nextml-model-data/codeformer/weights/facelib/detection_Resnet50_Final.pth": (
        "detection-resnet50-final.pth"
    ),
    "https://s3.eu-central-1.wasabisys.com/nextml-model-data/codeformer/weights/facelib/parsing_parsenet.pth": (
        "parsing-parsenet.pth"
    ),
}


def patch_not_impl():
    raise NotImplementedError()


def patch_cache_path(ctx: ServerContext, url: str, **kwargs) -> str:
    cache_path = cache_path_map.get(url, None)
    if cache_path is None:
        parsed = urlparse(url)
        cache_path = path.basename(parsed.path)

    cache_path = path.join(ctx.cache_path, cache_path)
    logger.debug("patching download path: %s -> %s", url, cache_path)

    if path.exists(cache_path):
        return cache_path
    else:
        raise FileNotFoundError("missing cache file: %s" % (cache_path))


def apply_patch_basicsr(ctx: ServerContext):
    logger.debug("patching BasicSR module")
    basicsr.utils.download_util.download_file_from_google_drive = patch_not_impl
    basicsr.utils.download_util.load_file_from_url = partial(patch_cache_path, ctx)


def apply_patch_codeformer(ctx: ServerContext):
    logger.debug("patching CodeFormer module")
    codeformer.facelib.utils.misc.download_pretrained_models = patch_not_impl
    codeformer.facelib.utils.misc.load_file_from_url = partial(patch_cache_path, ctx)


def apply_patch_facexlib(ctx: ServerContext):
    logger.debug("patching Facexlib module")
    facexlib.utils.load_file_from_url = partial(patch_cache_path, ctx)


def apply_patches(ctx: ServerContext):
    apply_patch_basicsr(ctx)
    apply_patch_codeformer(ctx)
    apply_patch_facexlib(ctx)
    unload(
        [
            "basicsr.utils.download_util",
            "codeformer.facelib.utils.misc",
            "facexlib.utils",
        ]
    )
