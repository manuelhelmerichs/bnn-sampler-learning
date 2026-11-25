"""Resolve the workspace without depending on the working directory."""

import tomllib
from pathlib import Path


def repository_root(start: Path | None = None) -> Path:
    origin = Path(start or __file__).resolve()
    for path in (origin, *origin.parents):
        project = path / "pyproject.toml"
        if project.is_file():
            with project.open("rb") as stream:
                config = tomllib.load(stream)
            if "workspace" in config.get("tool", {}).get("uv", {}):
                return path
    raise FileNotFoundError(f"No uv workspace above {origin}")


def relative(path: str | Path) -> str:
    """``path`` relative to the repository root, or absolute outside of it."""
    path, root = Path(path).resolve(), repository_root()
    return (path.relative_to(root) if path.is_relative_to(root) else path).as_posix()
