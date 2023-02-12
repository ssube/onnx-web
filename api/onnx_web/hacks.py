import sys
import codeformer.facelib.utils.misc
import basicsr.utils.download_util

from functools import partial
from logging import getLogger
from urllib.parse import urlparse
from os import path
from .utils import ServerContext, base_join

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
        pkg = mod.split('.', 1)[0]
        pkgs.append(pkg)

    to_unload = []
    for mod in sys.modules:
        if mod in exclude:
            continue

        if mod in pkgs:
            to_unload.append(mod)
            continue

        for pkg in pkgs:
            if mod.startswith(pkg + '.'):
                to_unload.append(mod)
                break

    logger.debug("Unloading modules for patching: %s", to_unload)
    for mod in to_unload:
        del sys.modules[mod]


def patch_not_impl():
    raise NotImplementedError()

def patch_cache_path(ctx: ServerContext, url: str, **kwargs) -> str:
    url = urlparse(url)
    base = path.basename(url.path)
    cache_path = path.join(ctx.model_path, ".cache", base)

    logger.debug("Patching download path: %s -> %s", path, cache_path)

    return cache_path

def apply_patch_codeformer(ctx: ServerContext):
    logger.debug("Patching CodeFormer module...")
    codeformer.facelib.utils.misc.download_pretrained_models = patch_not_impl
    codeformer.facelib.utils.misc.load_file_from_url = partial(patch_cache_path, ctx)

def apply_patch_basicsr(ctx: ServerContext):
    logger.debug("Patching BasicSR module...")
    basicsr.utils.download_util.download_file_from_google_drive = patch_not_impl
    basicsr.utils.download_util.load_file_from_url = partial(patch_cache_path, ctx)

def apply_patches(ctx: ServerContext):
    apply_patch_basicsr(ctx)
    apply_patch_codeformer(ctx)
    unload(["basicsr.utils.download_util", "codeformer.facelib.utils.misc"])
