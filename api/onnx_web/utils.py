from os import path
from typing import Any, Dict, Tuple


Border = Tuple[int, int, int, int]
Point = Tuple[int, int]
Size = Tuple[int, int]


def get_and_clamp_float(args, key: str, default_value: float, max_value: float, min_value=0.0) -> float:
    return min(max(float(args.get(key, default_value)), min_value), max_value)


def get_and_clamp_int(args, key: str, default_value: int, max_value: int, min_value=1) -> int:
    return min(max(int(args.get(key, default_value)), min_value), max_value)


def get_from_map(args, key: str, values: Dict[str, Any], default: Any):
    selected = args.get(key, default)
    if selected in values:
        return values[selected]
    else:
        return values[default]


def safer_join(base, tail):
    safer_path = path.relpath(path.normpath(path.join('/', tail)), '/')
    return path.join(base, safer_path)
