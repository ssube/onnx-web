from functools import cmp_to_key
from glob import glob
from logging import getLogger
from os import path
from typing import Dict, List, Union

import torch
import yaml

from ..image import (  # mask filters; noise sources
    mask_filter_gaussian_multiply,
    mask_filter_gaussian_screen,
    mask_filter_none,
    noise_source_fill_edge,
    noise_source_fill_mask,
    noise_source_gaussian,
    noise_source_histogram,
    noise_source_normal,
    noise_source_uniform,
)
from ..onnx.torch_before_ort import get_available_providers
from ..params import DeviceParams
from .context import ServerContext

logger = getLogger(__name__)

# config caching
config_params: Dict[str, Dict[str, Union[float, int, str]]] = {}

# pipeline params
platform_providers = {
    "cpu": "CPUExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "directml": "DmlExecutionProvider",
    "rocm": "ROCMExecutionProvider",
}
noise_sources = {
    "fill-edge": noise_source_fill_edge,
    "fill-mask": noise_source_fill_mask,
    "gaussian": noise_source_gaussian,
    "histogram": noise_source_histogram,
    "normal": noise_source_normal,
    "uniform": noise_source_uniform,
}
mask_filters = {
    "none": mask_filter_none,
    "gaussian-multiply": mask_filter_gaussian_multiply,
    "gaussian-screen": mask_filter_gaussian_screen,
}


# Available ORT providers
available_platforms: List[DeviceParams] = []

# loaded from model_path
correction_models: List[str] = []
diffusion_models: List[str] = []
inversion_models: List[str] = []
upscaling_models: List[str] = []


def get_config_params():
    return config_params


def get_available_platforms():
    return available_platforms


def get_correction_models():
    return correction_models


def get_diffusion_models():
    return diffusion_models


def get_inversion_models():
    return inversion_models


def get_upscaling_models():
    return upscaling_models


def get_mask_filters():
    return mask_filters


def get_noise_sources():
    return noise_sources


def get_config_value(key: str, subkey: str = "default", default=None):
    return config_params.get(key, {}).get(subkey, default)


def get_model_name(model: str) -> str:
    base = path.basename(model)
    (file, _ext) = path.splitext(base)
    return file


def load_models(context: ServerContext) -> None:
    global correction_models
    global diffusion_models
    global inversion_models
    global upscaling_models

    diffusion_models = [
        get_model_name(f) for f in glob(path.join(context.model_path, "diffusion-*"))
    ]
    diffusion_models.extend(
        [
            get_model_name(f)
            for f in glob(path.join(context.model_path, "stable-diffusion-*"))
        ]
    )
    diffusion_models = list(set(diffusion_models))
    diffusion_models.sort()

    correction_models = [
        get_model_name(f) for f in glob(path.join(context.model_path, "correction-*"))
    ]
    correction_models = list(set(correction_models))
    correction_models.sort()

    inversion_models = [
        get_model_name(f) for f in glob(path.join(context.model_path, "inversion-*"))
    ]
    inversion_models = list(set(inversion_models))
    inversion_models.sort()

    upscaling_models = [
        get_model_name(f) for f in glob(path.join(context.model_path, "upscaling-*"))
    ]
    upscaling_models = list(set(upscaling_models))
    upscaling_models.sort()


def load_params(context: ServerContext) -> None:
    global config_params
    params_file = path.join(context.params_path, "params.json")
    with open(params_file, "r") as f:
        config_params = yaml.safe_load(f)

        if "platform" in config_params and context.default_platform is not None:
            logger.info(
                "Overriding default platform from environment: %s",
                context.default_platform,
            )
            config_platform = config_params.get("platform", {})
            config_platform["default"] = context.default_platform


def load_platforms(context: ServerContext) -> None:
    global available_platforms

    providers = list(get_available_providers())

    for potential in platform_providers:
        if (
            platform_providers[potential] in providers
            and potential not in context.block_platforms
        ):
            if potential == "cuda":
                for i in range(torch.cuda.device_count()):
                    available_platforms.append(
                        DeviceParams(
                            potential,
                            platform_providers[potential],
                            {
                                "device_id": i,
                            },
                            context.optimizations,
                        )
                    )
            else:
                available_platforms.append(
                    DeviceParams(
                        potential,
                        platform_providers[potential],
                        None,
                        context.optimizations,
                    )
                )

    if context.any_platform:
        # the platform should be ignored when the job is scheduled, but set to CPU just in case
        available_platforms.append(
            DeviceParams(
                "any",
                platform_providers["cpu"],
                None,
                context.optimizations,
            )
        )

    # make sure CPU is last on the list
    def any_first_cpu_last(a: DeviceParams, b: DeviceParams):
        if a.device == b.device:
            return 0

        # any should be first, if it's available
        if a.device == "any":
            return -1

        # cpu should be last, if it's available
        if a.device == "cpu":
            return 1

        return -1

    available_platforms = sorted(
        available_platforms, key=cmp_to_key(any_first_cpu_last)
    )

    logger.info(
        "available acceleration platforms: %s",
        ", ".join([str(p) for p in available_platforms]),
    )
