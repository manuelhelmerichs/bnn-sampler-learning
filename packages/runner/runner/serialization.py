"""Canonical serialization and stable configuration identities."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np


def json_value(value: Any) -> Any:
    """Return a deterministic JSON-compatible representation of ``value``."""
    if dataclasses.is_dataclass(value):
        return json_value(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): json_value(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((json_value(item) for item in value), key=canonical_json)
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def canonical_json(value: Any) -> str:
    """Serialize with stable ordering and no insignificant whitespace."""
    return json.dumps(
        json_value(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def fingerprint(value: Any, *, namespace: str = "") -> str:
    """Identify configurations/selections, derive seeds, and deduplicate transforms.

    This is not a result checksum or a source-code invalidation mechanism.
    """
    payload = f"{namespace}\0{canonical_json(value)}".encode()
    return hashlib.sha256(payload).hexdigest()
