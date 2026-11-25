"""Which commit produced a result, and in which process environment.

The code of an experiment is every package, the experiment's own folder and the
lockfile; nothing else can change its measurements. Its record is the last
commit that changed that code, so commits of notes, reports, configurations or
other experiments leave it as it is. Configurations reach a result through the
fingerprints of the runs it read. Pilot and production results refuse
uncommitted code unless explicitly allowed, so the recorded commit always
contains the code that produced them.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from runner.paths import relative, repository_root

LIBRARIES = ("jax", "jaxlib", "numpy", "scipy", "flax", "blackjax", "optax", "zarr")


class DirtyWorkspaceError(RuntimeError):
    """The code has uncommitted changes."""


def _git(root: Path, *arguments: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def code_paths(module: str) -> tuple[str, ...]:
    """The paths whose content can change the measurements of an experiment."""
    folder = Path(importlib.import_module(module).__file__).parent
    return ("packages", relative(folder), "uv.lock")


def checkout(module: str) -> dict[str, Any]:
    """The last commit that changed the code of the experiment ``module``.

    ``dirty`` tells whether that code has changed since.
    """
    root, paths = repository_root(), code_paths(module)
    commit = _git(root, "log", "-1", "--format=%H", "--", *paths)
    if not commit:
        return {"commit": None, "dirty": None}
    status = _git(root, "status", "--porcelain", "--", *paths)
    return {"commit": commit, "dirty": None if status is None else bool(status)}


def require_clean(identity: dict[str, Any]) -> None:
    """Refuse code that the recorded commit would not describe."""
    if identity["dirty"]:
        raise DirtyWorkspaceError(
            "The code has uncommitted changes, so the recorded commit would not "
            "describe what produced the result. Commit first, or pass "
            "--allow-dirty to record the result as dirty."
        )


def runtime_provenance() -> dict[str, Any]:
    """Interpreter, platform, numerical libraries and backend of this process."""
    import jax

    libraries = {}
    for name in LIBRARIES:
        try:
            libraries[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "libraries": libraries,
        "backend": jax.default_backend(),
        "x64_enabled": bool(jax.config.jax_enable_x64),
        "environment": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "XLA_FLAGS")
        },
    }
