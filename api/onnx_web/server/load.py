from functools import cmp_to_key
from glob import glob
from logging import getLogger
from os import path
from typing import Any, Dict, List, Union

import torch
import yaml
from jsonschema import ValidationError, validate
from yaml import safe_load

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
from ..params import DeviceParams
from ..torch_before_ort import get_available_providers
from ..utils import merge
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

# Loaded from extra_models
extra_strings: Dict[str, Any] = {}


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


def get_extra_strings():
    return extra_strings


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


def load_extras(context: ServerContext):
    """
    Load the extras file(s) and collect the relevant parts for the server: labels and strings
    """
    global extra_strings

    labels = {}
    strings = {}

    with open("./schemas/extras.yaml", "r") as f:
        extra_schema = safe_load(f.read())

    for file in context.extra_models:
        if file is not None and file != "":
            logger.info("loading extra models from %s", file)
            try:
                with open(file, "r") as f:
                    data = safe_load(f.read())

                logger.debug("validating extras file %s", data)
                try:
                    validate(data, extra_schema)
                except ValidationError as err:
                    logger.error("invalid data in extras file: %s", err)
                    continue

                if "strings" in data:
                    logger.debug("collecting strings from %s", file)
                    merge(strings, data["strings"])

                for model_type in ["diffusion", "correction", "upscaling"]:
                    if model_type in data:
                        for model in data[model_type]:
                            if "label" in model:
                                model_name = model["name"]
                                logger.debug(
                                    "collecting label for model %s from %s",
                                    model_name,
                                    file,
                                )
                                labels[model_name] = model["label"]

                            if "inversions" in model:
                                for inversion in model["inversions"]:
                                    if "label" in inversion:
                                        inversion_name = inversion["name"]
                                        logger.debug(
                                            "collecting label for inversion %s from %s",
                                            inversion_name,
                                            model_name,
                                        )
                                        labels[
                                            f"inversion-{inversion_name}"
                                        ] = inversion["label"]

            except Exception as err:
                logger.error("error loading extras file: %s", err)

    logger.debug("adding labels to strings: %s", labels)
    merge(
        strings,
        {
            "en": {
                "translation": {
                    "model": labels,
                }
            }
        },
    )

    extra_strings = strings


def list_model_globs(context: ServerContext, globs: List[str]) -> List[str]:
    models = []
    for pattern in globs:
        pattern_path = path.join(context.model_path, pattern)
        logger.debug("loading models from %s", pattern_path)

        models.extend([get_model_name(f) for f in glob(pattern_path)])

    unique_models = list(set(models))
    unique_models.sort()
    return unique_models


def load_models(context: ServerContext) -> None:
    global correction_models
    global diffusion_models
    global inversion_models
    global upscaling_models

    diffusion_models = list_model_globs(
        context,
        [
            "diffusion-*",
            "stable-diffusion-*",
        ],
    )
    logger.debug("loaded diffusion models from disk: %s", diffusion_models)

    correction_models = list_model_globs(
        context,
        [
            "correction-*",
        ],
    )
    logger.debug("loaded correction models from disk: %s", correction_models)

    inversion_models = list_model_globs(
        context,
        [
            "inversion-*",
        ],
    )
    logger.debug("loaded inversion models from disk: %s", inversion_models)

    upscaling_models = list_model_globs(
        context,
        [
            "upscaling-*",
        ],
    )
    logger.debug("loaded upscaling models from disk: %s", upscaling_models)


def load_params(context: ServerContext) -> None:
    global config_params

    params_file = path.join(context.params_path, "params.json")
    logger.debug("loading server parameters from file: %s", params_file)

    with open(params_file, "r") as f:
        config_params = yaml.safe_load(f)

        if "platform" in config_params and context.default_platform is not None:
            logger.info(
                "overriding default platform from environment: %s",
                context.default_platform,
            )
            config_platform = config_params.get("platform", {})
            config_platform["default"] = context.default_platform


def load_platforms(context: ServerContext) -> None:
    global available_platforms

    providers = list(get_available_providers())
    logger.debug("loading available platforms from providers: %s", providers)

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
