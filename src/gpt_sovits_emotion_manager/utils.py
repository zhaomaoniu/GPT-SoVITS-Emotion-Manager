from typing import Any
from dataclasses import asdict, is_dataclass


def dump_dataclass(obj: Any) -> Any:
    """Recursively convert dataclass instances to dictionaries."""
    if is_dataclass(obj):
        return {k: dump_dataclass(v) for k, v in asdict(obj).items()}
    if isinstance(obj, (list, tuple)):
        return [dump_dataclass(i) for i in obj]
    if isinstance(obj, dict):
        return {k: dump_dataclass(v) for k, v in obj.items()}
    return obj
