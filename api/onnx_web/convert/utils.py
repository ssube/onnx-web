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

from ..constants import ONNX_WEIGHTS
from ..errors import RequestException
from ..server import ServerContext
from ..utils import get_boolean

logger = getLogger(__name__)

is_torch_2_0 = version.parse(
    version.parse(torch.__version__).base_version
) >= version.parse("2.0")


ModelDict = Dict[str, Union[str, int]]
LegacyModel = Tuple[str, str, Optional[bool], Optional[bool], Optional[int]]

DEFAULT_OPSET = 14
DIFFUSION_PREFIX = [
    "diffusion-",
    "diffusion/",
    "diffusion\\",
    "stable-diffusion-",
    "upscaling-",  # SD upscaling
]
MODEL_FORMATS = ["onnx", "pth", "ckpt", "safetensors"]
RESOLVE_FORMATS = ["safetensors", "ckpt", "pt", "pth", "bin"]


class ConversionContext(ServerContext):
    def __init__(
        self,
        model_path: str = ".",
        cache_path: Optional[str] = None,
        device: Optional[str] = None,
        half: bool = False,
        opset: int = DEFAULT_OPSET,
        token: Optional[str] = None,
        prune: Optional[List[str]] = None,
        control: bool = True,
        reload: bool = True,
        share_unet: bool = True,
        extract: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(model_path=model_path, cache_path=cache_path, **kwargs)

        self.control = control
        self.extract = extract
        self.half = half
        self.opset = opset
        self.prune = prune or []
        self.reload = reload
        self.share_unet = share_unet
        self.token = token

        if device is not None:
            self.training_device = device
        else:
            self.training_device = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def from_environ(cls):
        context = super().from_environ()
        context.control = get_boolean(environ, "ONNX_WEB_CONVERT_CONTROL", True)
        context.extract = get_boolean(environ, "ONNX_WEB_CONVERT_EXTRACT", False)
        context.reload = get_boolean(environ, "ONNX_WEB_CONVERT_RELOAD", True)
        context.share_unet = get_boolean(environ, "ONNX_WEB_CONVERT_SHARE_UNET", True)
        context.opset = int(environ.get("ONNX_WEB_CONVERT_OPSET", DEFAULT_OPSET))

        cpu_only = get_boolean(environ, "ONNX_WEB_CONVERT_CPU_ONLY", False)
        if cpu_only:
            context.training_device = "cpu"

        return context

    @property
    def map_location(self):
        return torch.device(self.training_device)


def download_progress(source: str, dest: str):
    dest_path = Path(dest).expanduser().resolve()
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if dest_path.exists():
        logger.debug("destination already exists: %s", dest_path)
        return str(dest_path.absolute())

    req = requests.get(
        source,
        stream=True,
        allow_redirects=True,
        headers={
            "User-Agent": "onnx-web-api",
        },
    )
    if req.status_code != 200:
        req.raise_for_status()  # Only works for 4xx errors, per SO answer
        raise RequestException(
            "request to %s failed with status code: %s" % (source, req.status_code)
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
        name, source, *_rest = model

        return {
            "name": name,
            "source": source,
        }
    else:
        return model


def tuple_to_correction(model: Union[ModelDict, LegacyModel]):
    if isinstance(model, list) or isinstance(model, tuple):
        name, source, *rest = model
        scale = rest.pop(0) if len(rest) > 0 else 1
        half = rest.pop(0) if len(rest) > 0 else False
        opset = rest.pop(0) if len(rest) > 0 else None

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
        single_vae = rest.pop(0) if len(rest) > 0 else False
        half = rest.pop(0) if len(rest) > 0 else False
        opset = rest.pop(0) if len(rest) > 0 else None

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
        scale = rest.pop(0) if len(rest) > 0 else 1
        half = rest.pop(0) if len(rest) > 0 else False
        opset = rest.pop(0) if len(rest) > 0 else None

        return {
            "name": name,
            "source": source,
            "half": half,
            "opset": opset,
            "scale": scale,
        }
    else:
        return model


def check_ext(name: str, exts: List[str]) -> Tuple[bool, str]:
    _name, ext = path.splitext(name)
    ext = ext.strip(".")

    return (ext in exts, ext)


def source_format(model: Dict) -> Optional[str]:
    if "format" in model:
        return model["format"]

    if "source" in model:
        valid, ext = check_ext(model["source"], MODEL_FORMATS)
        if valid:
            return ext

    return None


def remove_prefix(name: str, prefix: str) -> str:
    if name.startswith(prefix):
        return name[len(prefix) :]

    return name


def load_torch(name: str, map_location=None) -> Optional[Dict]:
    """
    TODO: move out of convert
    """
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
    """
    TODO: move out of convert
    """
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
            for next_extension in RESOLVE_FORMATS:
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

    if checkpoint is None:
        raise ValueError("error loading tensor")

    if checkpoint is not None and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    return checkpoint


def resolve_tensor(name: str) -> Optional[str]:
    """
    TODO: move out of convert
    """
    logger.debug("searching for tensors with known extensions: %s", name)
    for next_extension in RESOLVE_FORMATS:
        next_name = f"{name}.{next_extension}"
        if path.exists(next_name):
            return next_name

    return None


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
    v2=False,
    op_block_list=None,
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

    if v2 and op_block_list is None:
        op_block_list = ["Attention", "MultiHeadAttention"]

    if half:
        logger.info("converting model to fp16 internally: %s", output_file)
        infer_shapes_path(output_file)
        base_model = load_model(output_file)
        opt_model = convert_float_to_float16(
            base_model,
            disable_shape_infer=True,
            force_fp16_initializers=True,
            keep_io_types=True,
            op_block_list=op_block_list,
        )
        save_model(
            opt_model,
            f"{output_file}",
            save_as_external_data=external_data,
            all_tensors_to_one_file=True,
            location=ONNX_WEIGHTS,
        )


def fix_diffusion_name(name: str):
    if not any([name.startswith(prefix) for prefix in DIFFUSION_PREFIX]):
        logger.warning(
            "diffusion models must have names starting with diffusion- to be recognized by the server: %s does not match",
            name,
        )
        return f"diffusion-{name}"

    return name


def build_cache_paths(
    conversion: ConversionContext,
    name: str,
    client: Optional[str] = None,
    dest: Optional[str] = None,
    format: Optional[str] = None,
) -> List[str]:
    cache_path = dest or conversion.cache_path

    # add an extension if possible, some of the conversion code checks for it
    if format is not None:
        basename = path.basename(name)
        _filename, ext = path.splitext(basename)
        if ext is None or ext == "":
            name = f"{name}.{format}"

    paths = [
        path.join(cache_path, name),
    ]

    if client is not None:
        client_path = path.join(cache_path, client)
        paths.append(path.join(client_path, name))

    return paths


def get_first_exists(
    paths: List[str],
) -> Optional[str]:
    for name in paths:
        if path.exists(name):
            logger.debug("model already exists in cache, skipping fetch: %s", name)
            return name

    return None
