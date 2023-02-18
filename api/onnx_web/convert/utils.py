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
from yaml import safe_load

from ..utils import ServerContext

logger = getLogger(__name__)


ModelDict = Dict[str, Union[str, int]]
LegacyModel = Tuple[str, str, Optional[bool], Optional[bool], Optional[int]]


class ConversionContext(ServerContext):
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        cache_path: Optional[str] = None,
        half: Optional[bool] = False,
        opset: Optional[int] = None,
        token: Optional[str] = None,
    ) -> None:
        super().__init__(self, model_path=model_path, cache_path=cache_path)

        self.half = half
        self.opset = opset
        self.token = token

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
        ext = path.splitext(model["source"])
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


def load_yaml(file: str) -> str:
    with open(file, "r") as f:
        data = safe_load(f.read())
        return Config(data)


safe_chars = "._-"


def sanitize_name(name):
    return "".join(x for x in name if (x.isalnum() or x in safe_chars))


def remove_prefix(name, prefix):
    if name.startswith(prefix):
        return name[len(prefix) :]

    return name


def load_tensor(name: str, map_location=None):
    logger.info("loading model from checkpoint")
    _, extension = path.splitext(name)
    if extension.lower() == ".safetensors":
        environ["SAFETENSORS_FAST_GPU"] = "1"
        try:
            logger.debug("loading safetensors")
            checkpoint = safetensors.torch.load_file(name, device="cpu")
        except Exception as e:
            try:
                logger.warning(
                    "failed to load as safetensors file, falling back to torch", e
                )
                checkpoint = torch.jit.load(name)
            except Exception as e:
                logger.warning(
                    "failed to load with Torch JIT, falling back to PyTorch", e
                )
                checkpoint = torch.load(name, map_location=map_location)
                checkpoint = (
                    checkpoint["state_dict"]
                    if "state_dict" in checkpoint
                    else checkpoint
                )
    else:
        logger.debug("loading ckpt")
        checkpoint = torch.load(name, map_location=map_location)
        checkpoint = (
            checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        )

    return checkpoint
