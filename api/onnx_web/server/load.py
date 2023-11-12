from collections import defaultdict
from functools import cmp_to_key
from glob import glob
from logging import getLogger
from os import path
from typing import Any, Dict, List, Optional, Union

import torch
from jsonschema import ValidationError, validate

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
    source_filter_canny,
    source_filter_depth,
    source_filter_face,
    source_filter_gaussian,
    source_filter_hed,
    source_filter_mlsd,
    source_filter_noise,
    source_filter_none,
    source_filter_normal,
    source_filter_openpose,
    source_filter_scribble,
    source_filter_segment,
)
from ..models.meta import NetworkModel
from ..params import DeviceParams
from ..torch_before_ort import get_available_providers
from ..utils import load_config, merge
from .context import ServerContext

logger = getLogger(__name__)

# config caching
config_params: Dict[str, Dict[str, Union[float, int, str]]] = {}

# pipeline params
highres_methods = [
    "bilinear",
    "lanczos",
    "upscale",
]
mask_filters = {
    "none": mask_filter_none,
    "gaussian-multiply": mask_filter_gaussian_multiply,
    "gaussian-screen": mask_filter_gaussian_screen,
}
noise_sources = {
    "fill-edge": noise_source_fill_edge,
    "fill-mask": noise_source_fill_mask,
    "gaussian": noise_source_gaussian,
    "histogram": noise_source_histogram,
    "normal": noise_source_normal,
    "uniform": noise_source_uniform,
}
platform_providers = {
    "cpu": "CPUExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "directml": "DmlExecutionProvider",
    "rocm": "ROCMExecutionProvider",
    "tensorrt": "TensorRTExecutionProvider",
}
source_filters = {
    "canny": source_filter_canny,
    "depth": source_filter_depth,
    "face": source_filter_face,
    "gaussian": source_filter_gaussian,
    "hed": source_filter_hed,
    "mlsd": source_filter_mlsd,
    "noise": source_filter_noise,
    "none": source_filter_none,
    "normal": source_filter_normal,
    "openpose": source_filter_openpose,
    "segment": source_filter_segment,
    "scribble": source_filter_scribble,
}

# Available ORT providers
available_platforms: List[DeviceParams] = []

# loaded from model_path
correction_models: List[str] = []
diffusion_models: List[str] = []
network_models: List[NetworkModel] = []
upscaling_models: List[str] = []
wildcard_data: Dict[str, List[str]] = defaultdict(list)

# Loaded from extra_models
extra_hashes: Dict[str, str] = {}
extra_strings: Dict[str, Any] = {}
extra_tokens: Dict[str, List[str]] = {}


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


def get_wildcard_data():
    return wildcard_data


def get_extra_strings():
    return extra_strings


def get_extra_hashes():
    return extra_hashes


def get_highres_methods():
    return highres_methods


def get_mask_filters():
    return mask_filters


def get_noise_sources():
    return noise_sources


def get_source_filters():
    return source_filters


def get_config_value(key: str, subkey: str = "default", default=None):
    return config_params.get(key, {}).get(subkey, default)


def load_extras(server: ServerContext):
    """
    Load the extras file(s) and collect the relevant parts for the server: labels and strings
    """
    global extra_hashes
    global extra_strings
    global extra_tokens

    labels = {}
    strings = {}

    extra_schema = load_config("./schemas/extras.yaml")

    for file in server.extra_models:
        if file is not None and file != "":
            logger.info("loading extra models from %s", file)
            try:
                data = load_config(file)
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
                            model_name = model["name"]

                            if "hash" in model:
                                logger.debug(
                                    "collecting hash for model %s from %s",
                                    model_name,
                                    file,
                                )

                                extra_hashes[model_name] = model["hash"]

                            if "label" in model:
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

                            if "tokens" in model:
                                logger.debug("collecting tokens for model %s from %s", model_name, file)
                                extra_tokens[model_name] = model["tokens"]

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
    server: ServerContext,
    globs: List[str],
    base_path: Optional[str] = None,
    recursive=False,
    filename_only=True,
) -> List[str]:
    if base_path is None:
        base_path = server.model_path

    models = []
    for pattern in globs:
        pattern_path = path.join(base_path, pattern)
        logger.debug("loading models from %s", pattern_path)
        for name in glob(pattern_path, recursive=recursive):
            base = path.basename(name)
            (file, ext) = path.splitext(base)
            if ext not in IGNORE_EXTENSIONS:
                models.append(file if filename_only else path.relpath(name, base_path))

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
        [NetworkModel(model, "inversion", tokens=extra_tokens.get(model, [])) for model in inversion_models]
    )

    lora_models = list_model_globs(
        server,
        [
            "*",
        ],
        base_path=path.join(server.model_path, "lora"),
    )
    logger.debug("loaded LoRA models from disk: %s", lora_models)
    network_models.extend([NetworkModel(model, "lora", tokens=extra_tokens.get(model, [])) for model in lora_models])


def load_params(server: ServerContext) -> None:
    global config_params

    params_file = path.join(server.params_path, "params.json")
    logger.debug("loading server parameters from file: %s", params_file)

    config_params = load_config(params_file)

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


def load_wildcards(server: ServerContext) -> None:
    global wildcard_data

    wildcard_path = path.join(server.model_path, "wildcard")

    # simple wildcards
    wildcard_files = list_model_globs(
        server,
        ["**/*.txt"],
        base_path=wildcard_path,
        filename_only=False,
        recursive=True,
    )

    for file in wildcard_files:
        with open(
            path.join(server.model_path, "wildcard", file), "r", encoding="utf-8"
        ) as f:
            lines = f.read().splitlines()
            lines = [line.strip() for line in lines if not line.startswith("#")]
            lines = [line for line in lines if len(line) > 0]
            logger.debug("loading wildcards from %s: %s", file, lines)
            wildcard_data[path.splitext(file)[0]].extend(lines)

    structured_files = list_model_globs(
        server,
        ["**/*.json", "**/*.yaml"],
        base_path=wildcard_path,
        filename_only=False,
        recursive=True,
    )

    for file in structured_files:
        data = load_config(path.join(wildcard_path, file))
        logger.debug("loading structured wildcards from %s: %s", file, data)

        for key, values in data.items():
            if isinstance(values, list):
                wildcard_data[key].extend(values)
