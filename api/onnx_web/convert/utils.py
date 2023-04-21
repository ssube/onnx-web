import shutil
from functools import partial
from logging import getLogger
from os import environ, path
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import requests
import safetensors
import torch
from huggingface_hub.utils.tqdm import tqdm
from onnx import load_model, save_model
from onnx.shape_inference import infer_shapes_path
from onnxruntime.transformers.float16 import convert_float_to_float16
from packaging import version
from torch.onnx import export
from yaml import safe_load

from ..constants import ONNX_WEIGHTS
from ..server import ServerContext

logger = getLogger(__name__)

is_torch_2_0 = version.parse(
    version.parse(torch.__version__).base_version
) >= version.parse("2.0")


ModelDict = Dict[str, Union[str, int]]
LegacyModel = Tuple[str, str, Optional[bool], Optional[bool], Optional[int]]


class ConversionContext(ServerContext):
    def __init__(
        self,
        model_path: Optional[str] = None,
        cache_path: Optional[str] = None,
        device: Optional[str] = None,
        half: Optional[bool] = False,
        opset: Optional[int] = None,
        token: Optional[str] = None,
        prune: Optional[List[str]] = None,
        **kwargs,
    ) -> None:
        super().__init__(model_path=model_path, cache_path=cache_path, **kwargs)

        self.half = half
        self.opset = opset
        self.token = token
        self.prune = prune or []

        if device is not None:
            self.training_device = device
        else:
            self.training_device = "cuda" if torch.cuda.is_available() else "cpu"

        self.map_location = torch.device(self.training_device)


def download_progress(urls: List[Tuple[str, str]]):
    for url, dest in urls:
        dest_path = Path(dest).expanduser().resolve()
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        if dest_path.exists():
            logger.debug("destination already exists: %s", dest_path)
            return str(dest_path.absolute())

        req = requests.get(
            url,
            stream=True,
            allow_redirects=True,
            headers={
                "User-Agent": "onnx-web-api",
            },
        )
        if req.status_code != 200:
            req.raise_for_status()  # Only works for 4xx errors, per SO answer
            raise RuntimeError(
                "Request to %s failed with status code: %s" % (url, req.status_code)
            )

        total = int(req.headers.get("Content-Length", 0))
        desc = "unknown" if total == 0 else ""
        req.raw.read = partial(req.raw.read, decode_content=True)
        with tqdm.wrapattr(req.raw, "read", total=total, desc=desc) as data:
            with dest_path.open("wb") as f:
                shutil.copyfileobj(data, f)

        return str(dest_path.absolute())


def tuple_to_source(model: Union[ModelDict, LegacyModel]):
    if isinstance(model, list) or isinstance(model, tuple):
        name, source, *rest = model

        return {
            "name": name,
            "source": source,
        }
    else:
        return model


def tuple_to_correction(model: Union[ModelDict, LegacyModel]):
    if isinstance(model, list) or isinstance(model, tuple):
        name, source, *rest = model
        scale = rest[0] if len(rest) > 0 else 1
        half = rest[0] if len(rest) > 0 else False
        opset = rest[0] if len(rest) > 0 else None

        return {
            "name": name,
            "source": source,
            "half": half,
            "opset": opset,
            "scale": scale,
        }
    else:
        return model


def tuple_to_diffusion(model: Union[ModelDict, LegacyModel]):
    if isinstance(model, list) or isinstance(model, tuple):
        name, source, *rest = model
        single_vae = rest[0] if len(rest) > 0 else False
        half = rest[0] if len(rest) > 0 else False
        opset = rest[0] if len(rest) > 0 else None

        return {
            "name": name,
            "source": source,
            "half": half,
            "opset": opset,
            "single_vae": single_vae,
        }
    else:
        return model


def tuple_to_upscaling(model: Union[ModelDict, LegacyModel]):
    if isinstance(model, list) or isinstance(model, tuple):
        name, source, *rest = model
        scale = rest[0] if len(rest) > 0 else 1
        half = rest[0] if len(rest) > 0 else False
        opset = rest[0] if len(rest) > 0 else None

        return {
            "name": name,
            "source": source,
            "half": half,
            "opset": opset,
            "scale": scale,
        }
    else:
        return model


model_formats = ["onnx", "pth", "ckpt", "safetensors"]
model_formats_original = ["ckpt", "safetensors"]


def source_format(model: Dict) -> Optional[str]:
    if "format" in model:
        return model["format"]

    if "source" in model:
        _name, ext = path.splitext(model["source"])
        if ext in model_formats:
            return ext

    return None


class Config(object):
    """
    Shim for pydantic-style config.
    """

    def __init__(self, kwargs):
        self.__dict__.update(kwargs)
        for k, v in self.__dict__.items():
            Config.config_from_key(self, k, v)

    def __iter__(self):
        for k in self.__dict__.keys():
            yield k

    @classmethod
    def config_from_key(cls, target, k, v):
        if isinstance(v, dict):
            tmp = Config(v)
            setattr(target, k, tmp)
        else:
            setattr(target, k, v)


def load_yaml(file: str) -> Config:
    with open(file, "r") as f:
        data = safe_load(f.read())
        return Config(data)


def remove_prefix(name: str, prefix: str) -> str:
    if name.startswith(prefix):
        return name[len(prefix) :]

    return name


def load_torch(name: str, map_location=None) -> Optional[Dict]:
    try:
        logger.debug("loading tensor with Torch: %s", name)
        checkpoint = torch.load(name, map_location=map_location)
    except Exception:
        logger.exception(
            "error loading with Torch JIT, trying with Torch JIT: %s", name
        )
        checkpoint = torch.jit.load(name)

    return checkpoint


def load_tensor(name: str, map_location=None) -> Optional[Dict]:
    logger.debug("loading tensor: %s", name)
    _, extension = path.splitext(name)
    extension = extension[1:].lower()

    checkpoint = None
    if extension == "":
        # if no extension was intentional, do not search for others
        if path.exists(name):
            logger.debug("loading anonymous tensor")
            checkpoint = torch.load(name, map_location=map_location)
        else:
            logger.debug("searching for tensors with known extensions")
            for next_extension in ["safetensors", "ckpt", "pt", "bin"]:
                next_name = f"{name}.{next_extension}"
                if path.exists(next_name):
                    checkpoint = load_tensor(next_name, map_location=map_location)
                    if checkpoint is not None:
                        break
    elif extension == "safetensors":
        logger.debug("loading safetensors")
        try:
            environ["SAFETENSORS_FAST_GPU"] = "1"
            checkpoint = safetensors.torch.load_file(name, device="cpu")
        except Exception as e:
            logger.warning("error loading safetensor: %s", e)
    elif extension in ["bin", "ckpt", "pt"]:
        logger.debug("loading pickle tensor")
        try:
            checkpoint = load_torch(name, map_location=map_location)
        except Exception as e:
            logger.warning("error loading pickle tensor: %s", e)
    elif extension in ["onnx", "pt"]:
        logger.warning(
            "tensor has ONNX extension, attempting to use PyTorch anyways: %s",
            extension,
        )
        try:
            checkpoint = load_torch(name, map_location=map_location)
        except Exception as e:
            logger.warning("error loading tensor: %s", e)
    else:
        logger.warning("unknown tensor type, falling back to PyTorch: %s", extension)
        try:
            checkpoint = load_torch(name, map_location=map_location)
        except Exception as e:
            logger.warning("error loading tensor: %s", e)

    if checkpoint is not None and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    return checkpoint


def onnx_export(
    model,
    model_args: tuple,
    output_path: Path,
    ordered_input_names,
    output_names,
    dynamic_axes,
    opset,
    half=False,
    external_data=False,
):
    """
    From https://github.com/huggingface/diffusers/blob/main/scripts/convert_stable_diffusion_checkpoint_to_onnx.py
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_file = output_path.absolute().as_posix()

    export(
        model,
        model_args,
        f=output_file,
        input_names=ordered_input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        opset_version=opset,
    )

    if half:
        logger.info("converting model to fp16 internally: %s", output_file)
        infer_shapes_path(output_file)
        base_model = load_model(output_file)
        opt_model = convert_float_to_float16(
            base_model,
            disable_shape_infer=True,
            keep_io_types=True,
            force_fp16_initializers=True,
        )
        save_model(
            opt_model,
            f"{output_file}",
            save_as_external_data=external_data,
            all_tensors_to_one_file=True,
            location=ONNX_WEIGHTS,
        )
