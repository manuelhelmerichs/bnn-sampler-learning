"""Named seed streams that do not depend on process order."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from runner.serialization import fingerprint


def derive_seed(root_seed: int, stream: str, factors: Mapping[str, Any]) -> int:
    """Derive a stable, named 32-bit seed without depending on process order."""
    digest = fingerprint(
        {"root_seed": root_seed, "stream": stream, "factors": factors},
        namespace="seed-stream",
    )
    return int(digest[:8], 16)
