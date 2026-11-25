"""Which code and environment produced a run.

A run records the git commit of the workspace and a content hash of the code
it can import. That code is every module of this package that the command line
or a worker reaches; the rest of the package, such as reference targets, never
runs in a run and may change while one is under way. Starting from uncommitted
code is refused unless explicitly allowed, so the recorded commit always
contains the code that ran.
"""

from __future__ import annotations

import ast
import functools
import hashlib
import importlib.metadata
import importlib.util
import platform
import subprocess
from collections.abc import Iterable
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
ENTRY_POINTS = (f"{PACKAGE.name}.__main__", f"{PACKAGE.name}.pipeline.worker")
LIBRARIES = ("jax", "jaxlib", "numpy", "scipy", "flax", "blackjax", "optax", "zarr")


class DirtyWorkspaceError(RuntimeError):
    """The code of a run or the lockfile has uncommitted changes."""


def file_digest(path: str | Path) -> str:
    """SHA-256 of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _named_modules(path: Path, module: str, modules: dict[str, Path]) -> set[str]:
    """Modules of this package that a source file imports or names."""
    package = module if path.name == "__init__.py" else module.rpartition(".")[0]
    names = set()
    for node in ast.walk(ast.parse(path.read_bytes())):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            source = node.module or ""
            if node.level:
                source = importlib.util.resolve_name("." * node.level + source, package)
            names.add(source)
            names.update(f"{source}.{alias.name}" for alias in node.names)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            names.add(node.value)
    found = set()
    for name in names & modules.keys():
        # Importing a module also runs the packages that contain it.
        parts = name.split(".")
        found.update(".".join(parts[:end]) for end in range(1, len(parts) + 1))
    return found


@functools.cache
def run_sources(imports: tuple[str, ...] = ()) -> tuple[Path, ...]:
    """Source files of this package that a run can import, in a stable order.

    A module counts when an entry point reaches it through import statements
    at any depth, or through string literals that name a module, which is how
    lazily imported components are listed. A module name that is assembled at
    run time is not seen, so name every lazily imported module in a literal.
    ``imports`` are the extension modules of a configuration; those that
    belong to this package are reached as well. The answer is kept for the
    process, whose code no longer changes once it is imported.
    """
    modules = {}
    for path in PACKAGE.rglob("*.py"):
        parts = (PACKAGE.name, *path.relative_to(PACKAGE).with_suffix("").parts)
        modules[".".join(parts[:-1] if parts[-1] == "__init__" else parts)] = path
    pending = [name for name in (*ENTRY_POINTS, *imports) if name in modules]
    reached: set[str] = set()
    while pending:
        module = pending.pop()
        if module not in reached:
            reached.add(module)
            pending.extend(_named_modules(modules[module], module, modules) - reached)
    return tuple(sorted(modules[module] for module in reached))


def source_digest(sources: Iterable[Path]) -> str:
    """Content hash of source files of this package."""
    digest = hashlib.sha256()
    for path in sources:
        digest.update(path.relative_to(PACKAGE).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _git(*arguments: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=PACKAGE,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def code_identity(imports: Iterable[str] = ()) -> dict:
    """Commit, cleanliness and content of the code of a run and its environment.

    ``imports`` are the extension modules of the run's configuration.
    """
    sources = run_sources(tuple(imports))
    root = _git("rev-parse", "--show-toplevel")
    commit = dirty = lock = None
    if root is not None:
        commit = _git("rev-parse", "HEAD")
        lockfile = Path(root) / "uv.lock"
        status = _git("status", "--porcelain", "--", *map(str, sources), str(lockfile))
        dirty = bool(status)
        lock = file_digest(lockfile) if lockfile.is_file() else None
    versions = {}
    for name in LIBRARIES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {
        "commit": commit,
        "dirty": dirty,
        "source_sha256": source_digest(sources),
        "lock_sha256": lock,
        "python": platform.python_version(),
        "libraries": versions,
    }


def require_clean(identity: dict) -> None:
    """Refuse code whose recorded commit would not contain what is running."""
    if identity["dirty"]:
        raise DirtyWorkspaceError(
            "The code of a run or uv.lock has uncommitted changes, so the commit "
            "stamped on this run would not describe the code that produced it. "
            "Commit first, or pass --allow-dirty to record the run as dirty."
        )
