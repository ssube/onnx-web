from functools import cmp_to_key
from glob import glob
from logging import getLogger
from os import path
from typing import Any, Dict, List, Optional, Union

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
from ..models.meta import NetworkModel
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
highres_methods = [
    "bilinear",
    "lanczos",
    "upscale",
]


# Available ORT providers
available_platforms: List[DeviceParams] = []

# loaded from model_path
correction_models: List[str] = []
diffusion_models: List[str] = []
network_models: List[NetworkModel] = []
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


def get_network_models():
    return network_models


def get_upscaling_models():
    return upscaling_models


def get_extra_strings():
    return extra_strings


def get_highres_methods():
    return highres_methods


def get_mask_filters():
    return mask_filters


def get_noise_sources():
    return noise_sources


def get_config_value(key: str, subkey: str = "default", default=None):
    return config_params.get(key, {}).get(subkey, default)


def load_extras(server: ServerContext):
    """
    Load the extras file(s) and collect the relevant parts for the server: labels and strings
    """
    global extra_strings

    labels = {}
    strings = {}

    with open("./schemas/extras.yaml", "r") as f:
        extra_schema = safe_load(f.read())

    for file in server.extra_models:
        if file is not None and file != "":
            logger.info("loading extra models from %s", file)
            try:
                with open(file, "r") as f:
                    data = safe_load(f.read())

                logger.debug("validating extras file %s", data)
                try:
                    validate(data, extra_schema)
                except ValidationError:
                    logger.exception("invalid data in extras file")
                    continue

                if "strings" in data:
                    logger.debug("collecting strings from %s", file)
                    merge(strings, data["strings"])

                for model_type in ["diffusion", "correction", "upscaling", "networks"]:
                    if model_type in data:
                        for model in data[model_type]:
                            if "label" in model:
                                model_name = model["name"]
                                logger.debug(
                                    "collecting label for model %s from %s",
                                    model_name,
                                    file,
                                )

                                if "type" in model:
                                    labels[f'{model["type"]}.{model_name}'] = model[
                                        "label"
                                    ]
                                else:
                                    labels[model_name] = model["label"]

                            if "inversions" in model:
                                for inversion in model["inversions"]:
                                    if "label" in inversion:
                                        inversion_name = inversion["name"]
                                        logger.debug(
                                            "collecting label for Textual Inversion %s from %s",
                                            inversion_name,
                                            model_name,
                                        )
                                        labels[
                                            f"inversion.{inversion_name}"
                                        ] = inversion["label"]

                            if "loras" in model:
                                for lora in model["loras"]:
                                    if "label" in lora:
                                        lora_name = lora["name"]
                                        logger.debug(
                                            "collecting label for LoRA %s from %s",
                                            lora_name,
                                            model_name,
                                        )
                                        labels[f"lora.{lora_name}"] = lora["label"]

            except Exception:
                logger.exception("error loading extras file")

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


IGNORE_EXTENSIONS = [".crdownload", ".lock", ".tmp"]


def list_model_globs(
    server: ServerContext, globs: List[str], base_path: Optional[str] = None
) -> List[str]:
    models = []
    for pattern in globs:
        pattern_path = path.join(base_path or server.model_path, pattern)
        logger.debug("loading models from %s", pattern_path)
        for name in glob(pattern_path):
            base = path.basename(name)
            (file, ext) = path.splitext(base)
            if ext not in IGNORE_EXTENSIONS:
                models.append(file)

    unique_models = list(set(models))
    unique_models.sort()
    return unique_models


def load_models(server: ServerContext) -> None:
    global correction_models
    global diffusion_models
    global network_models
    global upscaling_models

    # main categories
    diffusion_models = list_model_globs(
        server,
        [
            "diffusion-*",
            "stable-diffusion-*",
        ],
    )
    diffusion_models.extend(
        list_model_globs(
            server,
            ["*"],
            base_path=path.join(server.model_path, "diffusion"),
        )
    )
    logger.debug("loaded diffusion models from disk: %s", diffusion_models)

    correction_models = list_model_globs(
        server,
        [
            "correction-*",
        ],
    )
    correction_models.extend(
        list_model_globs(
            server,
            ["*"],
            base_path=path.join(server.model_path, "correction"),
        )
    )
    logger.debug("loaded correction models from disk: %s", correction_models)

    upscaling_models = list_model_globs(
        server,
        [
            "upscaling-*",
        ],
    )
    upscaling_models.extend(
        list_model_globs(
            server,
            ["*"],
            base_path=path.join(server.model_path, "upscaling"),
        )
    )
    logger.debug("loaded upscaling models from disk: %s", upscaling_models)

    # additional networks
    control_models = list_model_globs(
        server,
        [
            "*",
        ],
        base_path=path.join(server.model_path, "control"),
    )
    logger.debug("loaded ControlNet models from disk: %s", control_models)
    network_models.extend([NetworkModel(model, "control") for model in control_models])

    inversion_models = list_model_globs(
        server,
        [
            "*",
        ],
        base_path=path.join(server.model_path, "inversion"),
    )
    logger.debug("loaded Textual Inversion models from disk: %s", inversion_models)
    network_models.extend(
        [NetworkModel(model, "inversion") for model in inversion_models]
    )

    lora_models = list_model_globs(
        server,
        [
            "*",
        ],
        base_path=path.join(server.model_path, "lora"),
    )
    logger.debug("loaded LoRA models from disk: %s", lora_models)
    network_models.extend([NetworkModel(model, "lora") for model in lora_models])


def load_params(server: ServerContext) -> None:
    global config_params

    params_file = path.join(server.params_path, "params.json")
    logger.debug("loading server parameters from file: %s", params_file)

    with open(params_file, "r") as f:
        config_params = yaml.safe_load(f)

        if "platform" in config_params and server.default_platform is not None:
            logger.info(
                "overriding default platform from environment: %s",
                server.default_platform,
            )
            config_platform = config_params.get("platform", {})
            config_platform["default"] = server.default_platform


def load_platforms(server: ServerContext) -> None:
    global available_platforms

    providers = list(get_available_providers())
    logger.debug("loading available platforms from providers: %s", providers)

    for potential in platform_providers:
        if (
            platform_providers[potential] in providers
            and potential not in server.block_platforms
        ):
            if potential == "cuda" or potential == "rocm":
                for i in range(torch.cuda.device_count()):
                    options = {
                        "device_id": i,
                    }

                    if potential == "cuda" and server.memory_limit is not None:
                        options["arena_extend_strategy"] = "kSameAsRequested"
                        options["gpu_mem_limit"] = server.memory_limit

                    available_platforms.append(
                        DeviceParams(
                            potential,
                            platform_providers[potential],
                            options,
                            server.optimizations,
                        )
                    )
            else:
                available_platforms.append(
                    DeviceParams(
                        potential,
                        platform_providers[potential],
                        None,
                        server.optimizations,
                    )
                )

    if server.any_platform:
        # the platform should be ignored when the job is scheduled, but set to CPU just in case
        available_platforms.append(
            DeviceParams(
                "any",
                platform_providers["cpu"],
                None,
                server.optimizations,
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
