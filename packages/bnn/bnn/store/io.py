"""Atomic, durable file primitives shared by every artifact.

A reader observes either the previous complete file or the next complete file.
Advisory locks coordinate the processes of one machine or one shared POSIX
file system.
"""

from __future__ import annotations

import fcntl
import json
import os
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, BinaryIO


def sync_directory(path: str | Path) -> None:
    """Persist the entries of a directory."""
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write(path: str | Path, write: Callable[[BinaryIO], None]) -> None:
    """Write through a temporary sibling and publish it with one rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as file:
            write(file)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
        sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path: str | Path, value: Any) -> None:
    """Atomically write finite, key-sorted JSON."""
    encoded = json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode()
    atomic_write(path, lambda file: file.write(encoded + b"\n"))


def read_json(path: str | Path) -> Any:
    """Read a JSON document."""
    return json.loads(Path(path).read_text())


@contextmanager
def lock(path: str | Path, *, blocking: bool = True) -> Iterator[None]:
    """Hold an exclusive advisory lock; ``BlockingIOError`` if busy and nonblocking."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        yield
    finally:
        os.close(descriptor)
