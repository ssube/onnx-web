import importlib
import json
from functools import reduce
from hashlib import sha256
from json import JSONDecodeError
from logging import getLogger
from os import environ, path
from platform import system
from struct import pack
from typing import Any, Dict, List, Optional, Sequence, TypeVar

from yaml import safe_load

logger = getLogger(__name__)

SAFE_CHARS = "._-"


def split_list(val: str) -> List[str]:
    parts = [part.strip() for part in val.split(",")]
    return [part for part in parts if len(part) > 0]


def base_join(base: str, tail: str) -> str:
    tail_path = path.relpath(path.normpath(path.join("/", tail)), "/")
    return path.join(base, tail_path)


def is_debug() -> bool:
    return get_boolean(environ, "DEBUG", False)


def recursive_get(d, keys, default_value=None):
    empty_dict = {}
    val = reduce(lambda c, k: c.get(k, empty_dict), keys, d)

    if val == empty_dict:
        return default_value

    return val


def get_boolean(args: Any, key: str, default_value: bool) -> bool:
    val = recursive_get(args, key.split("."), default_value=str(default_value))

    if isinstance(val, bool):
        return val

    return val.lower() in ("1", "t", "true", "y", "yes")


def get_list(args: Any, key: str, default="") -> List[str]:
    val = recursive_get(args, key.split("."), default=default)
    return split_list(val)


def get_and_clamp_float(
    args: Any, key: str, default_value: float, max_value: float, min_value=0.0
) -> float:
    val = recursive_get(args, key.split("."), default=default_value)
    return min(max(float(val), min_value), max_value)


def get_and_clamp_int(
    args: Any, key: str, default_value: int, max_value: int, min_value=1
) -> int:
    val = recursive_get(args, key.split("."), default=default_value)
    return min(max(int(val), min_value), max_value)


TElem = TypeVar("TElem")


def get_from_list(
    args: Any, key: str, values: Sequence[TElem], default_value: Optional[TElem] = None
) -> Optional[TElem]:
    selected = args.get(key, default_value)
    if selected in values:
        return selected

    logger.warning("invalid selection %s, options: %s", selected, values)
    if len(values) > 0:
        return values[0]

    return None


def get_from_map(
    args: Any, key: str, values: Dict[str, TElem], default_key: str
) -> TElem:
    selected = args.get(key, default_key)
    if selected in values:
        return values[selected]
    else:
        return values[default_key]


def get_not_empty(args: Any, key: str, default: TElem) -> TElem:
    val = args.get(key, default)

    if val is None or len(val) == 0:
        val = default

    return val


def run_gc(devices: Optional[List[Any]] = None):
    """
    Deprecated, use `onnx_web.device.run_gc` instead.
    """
    from .device import run_gc as run_gc_impl

    logger.debug("calling deprecated run_gc, please use onnx_web.device.run_gc instead")
    run_gc_impl(devices)


def sanitize_name(name):
    return "".join(x for x in name if (x.isalnum() or x in SAFE_CHARS))


def merge(a, b, path=None):
    "merges b into a"
    if path is None:
        path = []
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                merge(a[key], b[key], path + [str(key)])
            elif a[key] == b[key]:
                pass  # same leaf value
            else:
                raise ValueError("conflict at %s" % ".".join(path + [str(key)]))
        else:
            a[key] = b[key]
    return a


toaster = None


def show_system_toast(msg: str) -> None:
    global toaster

    sys_name = system()
    if sys_name == "Linux":
        if (
            importlib.util.find_spec("gi") is not None
            and importlib.util.find_spec("gi.repository") is not None
        ):
            from gi.repository import Notify

            if toaster is None:
                Notify.init("onnx-web")

            Notify.Notification.new(msg).show()
        else:
            logger.info(
                "please install the PyGObject module to enable toast notifications on Linux"
            )
    elif sys_name == "Windows":
        if importlib.util.find_spec("win10toast") is not None:
            from win10toast import ToastNotifier

            if toaster is None:
                toaster = ToastNotifier()

            toaster.show_toast(msg, duration=15)
        else:
            logger.info(
                "please install the win10toast module to enable toast notifications on Windows"
            )
    else:
        logger.info("system notifications not yet available for %s", sys_name)


def load_json(file: str) -> Dict:
    with open(file, "r") as f:
        data = json.loads(f.read())
        return data


def load_yaml(file: str) -> Dict:
    with open(file, "r") as f:
        data = safe_load(f.read())
        return data


def load_config(file: str) -> Dict:
    name, ext = path.splitext(file)
    if ext in [".yml", ".yaml"]:
        return load_yaml(file)
    elif ext in [".json"]:
        return load_json(file)
    else:
        raise ValueError("unknown config file extension")


def load_config_str(raw: str) -> Dict:
    try:
        return json.loads(raw)
    except JSONDecodeError:
        return safe_load(raw)


HASH_BUFFER_SIZE = 2**22  # 4MB


def hash_file(name: str):
    sha = sha256()
    with open(name, "rb") as f:
        while True:
            data = f.read(HASH_BUFFER_SIZE)
            if not data:
                break

            sha.update(data)

    return sha.hexdigest()


def hash_value(sha, param: Optional[Any]):
    if param is None:
        return None
    elif isinstance(param, bool):
        sha.update(bytearray(pack("!B", param)))
    elif isinstance(param, float):
        sha.update(bytearray(pack("!f", param)))
    elif isinstance(param, int):
        sha.update(bytearray(pack("!I", param)))
    elif isinstance(param, str):
        sha.update(param.encode("utf-8"))
    else:
        logger.warning("cannot hash param: %s, %s", param, type(param))


def coalesce(*args, throw=False):
    for arg in args:
        if arg is not None:
            return arg

    if throw:
        raise ValueError("no value found")

    return None
