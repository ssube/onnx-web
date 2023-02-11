import shutil
from functools import partial
from logging import getLogger
from os import path
from pathlib import Path
from typing import Dict, Union, List, Optional, Tuple

import requests
import torch
from tqdm.auto import tqdm

logger = getLogger(__name__)


ModelDict = Dict[str, Union[str, int]]
LegacyModel = Tuple[str, str, Optional[bool], Optional[bool], Optional[int]]


class ConversionContext:
    def __init__(
        self,
        model_path: str,
        device: str,
        cache_path: Optional[str] = None,
        half: Optional[bool] = False,
        opset: Optional[int] = None,
        token: Optional[str] = None,
    ) -> None:
        self.model_path = model_path
        self.cache_path = cache_path or path.join(model_path, ".cache")
        self.training_device = device
        self.map_location = torch.device(device)
        self.half = half
        self.opset = opset
        self.token = token


def download_progress(urls: List[Tuple[str, str]]):
    for url, dest in urls:
        dest_path = Path(dest).expanduser().resolve()
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        if dest_path.exists():
            logger.info("Destination already exists: %s", dest_path)
            return str(dest_path.absolute())

        req = requests.get(url, stream=True, allow_redirects=True)
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


known_formats = ["onnx", "pth", "ckpt", "safetensors"]

def source_format(model: Dict) -> Optional[str]:
    if "format" in model:
        return model["format"]

    if "source" in model:
        ext = path.splitext(model["source"])
        if ext in known_formats:
            return ext

    return None
