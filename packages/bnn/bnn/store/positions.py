"""Append-only posterior draws in immutable, sharded Zarr v3 blocks.

A store holds one sequence of draws per chain. Every append publishes one
immutable block and then atomically advances the chain's head, so a reader
never observes a partial block and a crash never corrupts committed draws. A
head record identifies an exact prefix of a chain; ``restore_head`` returns a
chain to such a prefix, which lets a caller commit draws together with its own
state.

Only numeric payloads live in Zarr. A declarative schema describes the tree and
the exact byte representation of each leaf, including bfloat16.
"""

from __future__ import annotations

import bisect
import hashlib
import math
import os
import shutil
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import zarr
from zarr.codecs import Crc32cCodec, ZstdCodec

from bnn.store.io import lock, read_json, sync_directory, write_json
from bnn.store.trees import (
    decode_structure,
    is_storable,
    structure_of,
    to_device,
)

SCHEMA = "schema.json"
CHAINS = "chains.json"
_EMPTY_DIGEST = hashlib.sha256(b"").hexdigest()
EMPTY_HEAD = {"block": None, "count": 0, "last_draw": -1, "digest": _EMPTY_DIGEST}
CHUNK_BYTES = 1024**2
SHARD_BYTES = 16 * 1024**2


def _chain_path(directory: Path, chain: int) -> Path:
    return Path(directory) / "chains" / str(int(chain))


def _schema_for(tree: Any) -> dict:
    """Byte layout of one draw; ``tree`` has a leading draw axis on every leaf."""
    specs, offset = [], 0
    for leaf in jax.tree.leaves(tree):
        value = np.asarray(leaf)
        if not is_storable(value.dtype):
            raise TypeError(f"Unsupported posterior dtype: {value.dtype}.")
        shape = value.shape[1:]
        nbytes = math.prod(shape) * value.dtype.itemsize
        specs.append(
            {
                "shape": list(shape),
                "dtype": value.dtype.str,
                "logical_dtype": str(value.dtype),
                "offset": offset,
                "nbytes": nbytes,
            }
        )
        offset += nbytes
    single = jax.tree.map(lambda leaf: 0, tree)
    return {
        "structure": structure_of(single),
        "leaves": specs,
        "bytes_per_draw": offset,
    }


def _dtype(spec: dict) -> np.dtype:
    if spec["logical_dtype"] == "bfloat16":
        return np.dtype(jnp.bfloat16)
    return np.dtype(spec["dtype"])


def _ensure_schema(directory: Path, schema: dict) -> None:
    with lock(directory / ".schema.lock"):
        path = directory / SCHEMA
        if not path.exists():
            write_json(path, schema)
        elif read_json(path) != schema:
            raise ValueError(
                f"Draws do not match the tree, shapes or dtypes of the store: {directory}"
            )


def _geometry(rows, columns, itemsize, chunk_bytes, shard_bytes):
    columns = max(1, columns)
    width = min(columns, max(1, chunk_bytes // itemsize))
    height = min(rows, max(1, chunk_bytes // (width * itemsize)))
    shard_rows = height * max(
        1, min(math.ceil(rows / height), shard_bytes // (height * width * itemsize))
    )
    shard_width = width * max(
        1,
        min(math.ceil(columns / width), shard_bytes // (shard_rows * width * itemsize)),
    )
    return (height, width), (shard_rows, shard_width)


def _create_array(group, name, data, chunk_bytes, shard_bytes):
    chunks, shards = _geometry(
        data.shape[0], data.shape[1], data.dtype.itemsize, chunk_bytes, shard_bytes
    )
    group.create_array(
        name,
        data=data,
        chunks=chunks,
        shards=shards,
        compressors=[ZstdCodec(level=1), Crc32cCodec()],
        config={"write_empty_chunks": True},
    )
    return {
        "shape": list(data.shape),
        "chunks": list(chunks),
        "shards": list(shards),
        "dtype": data.dtype.str,
    }


def _seal(directory: Path) -> dict:
    """Persist a block and record its files, so a lost shard cannot read as zeros."""
    files, directories = {}, []
    for path in directory.rglob("*"):
        if path.is_dir():
            directories.append(path)
        elif path.is_file():
            with path.open("rb") as file:
                os.fsync(file.fileno())
            spec = {"size": path.stat().st_size}
            if path.name == "zarr.json":
                spec["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            files[str(path.relative_to(directory))] = spec
    for path in sorted(directories, key=lambda p: len(p.parts), reverse=True):
        sync_directory(path)
    sync_directory(directory)
    return files


def head(directory: str | Path, chain: int) -> dict:
    """The committed prefix of a chain: last block, draw count, last ID, ID digest."""
    path = _chain_path(Path(directory), chain) / "head.json"
    return read_json(path) if path.exists() else dict(EMPTY_HEAD)


def restore_head(directory: str | Path, chain: int, record: dict) -> None:
    """Return a chain to an earlier committed prefix and drop the later blocks."""
    directory = Path(directory)
    chain_path = _chain_path(directory, chain)
    with lock(chain_path / ".writer.lock"):
        current = head(directory, chain)
        if current == record:
            return
        if record["count"] > current["count"]:
            raise ValueError("Cannot restore a chain to draws it never committed.")
        reachable, block = set(), record["block"]
        while block is not None:
            reachable.add(block)
            block = read_json(chain_path / "blocks" / block / "commit.json")["previous"]
        if record["block"] is None:
            (chain_path / "head.json").unlink(missing_ok=True)
        else:
            write_json(chain_path / "head.json", record)
        blocks = chain_path / "blocks"
        if blocks.exists():
            for path in blocks.iterdir():
                if path.name not in reachable:
                    shutil.rmtree(path, ignore_errors=True)
            sync_directory(blocks)


def truncate(directory: str | Path, chain: int, count: int) -> None:
    """Drop the last blocks of a chain until exactly ``count`` draws remain."""
    directory = Path(directory)
    record = head(directory, chain)
    while record["count"] > count:
        commit = read_json(
            _chain_path(directory, chain) / "blocks" / record["block"] / "commit.json"
        )
        if commit["previous"] is None:
            record = dict(EMPTY_HEAD)
        else:
            previous = read_json(
                _chain_path(directory, chain)
                / "blocks"
                / commit["previous"]
                / "commit.json"
            )
            record = {
                "block": commit["previous"],
                "count": commit["start"],
                "last_draw": previous["last_draw"],
                "digest": previous["digest"],
            }
    if record["count"] != count:
        raise ValueError(f"No block of chain {chain} ends after {count} draws.")
    restore_head(directory, chain, record)


def append_block(
    directory: str | Path,
    chain: int,
    draw_ids: Iterable[int],
    tree: Any,
    *,
    chunk_bytes: int = CHUNK_BYTES,
    shard_bytes: int = SHARD_BYTES,
) -> dict:
    """Commit consecutive draws of one chain as one block; return the new head.

    Every leaf of ``tree`` has the draws on its leading axis. Draw IDs increase
    strictly within a chain.
    """
    directory = Path(directory)
    ids = np.asarray(
        list(draw_ids) if not isinstance(draw_ids, np.ndarray) else draw_ids
    )
    if ids.ndim != 1 or ids.dtype.kind not in "iu" or (ids.size and ids.min() < 0):
        raise ValueError("Draw IDs must be a vector of nonnegative integers.")
    leaves = [np.asarray(leaf) for leaf in jax.tree.leaves(tree)]
    if any(leaf.ndim < 1 or len(leaf) != len(ids) for leaf in leaves):
        raise ValueError("Every leaf needs one leading entry per draw ID.")
    chain = int(chain)
    chain_path = _chain_path(directory, chain)
    with lock(chain_path / ".writer.lock"):
        current = head(directory, chain)
        if not len(ids):
            return current
        if ids[0] <= current["last_draw"] or np.any(np.diff(ids) <= 0):
            raise ValueError("Draw IDs must increase strictly within a chain.")
        schema = _schema_for(tree)
        _ensure_schema(directory, schema)
        if schema["bytes_per_draw"]:
            values = np.concatenate(
                [
                    np.ascontiguousarray(leaf).reshape(len(ids), -1).view(np.uint8)
                    for leaf in leaves
                    if leaf[0].size
                ],
                axis=1,
            )
        else:
            # A tree without entries still stores its draw axis.
            values = np.zeros((len(ids), 1), dtype=np.uint8)
        digest = current["digest"]
        for draw in ids.tolist():
            digest = hashlib.sha256(
                bytes.fromhex(digest) + int(draw).to_bytes(8, "little")
            ).hexdigest()
        blocks = chain_path / "blocks"
        blocks.mkdir(parents=True, exist_ok=True)
        block_id = uuid.uuid4().hex
        temporary, destination = blocks / f".{block_id}.pending", blocks / block_id
        try:
            group = zarr.open_group(str(temporary), mode="w", zarr_format=3)
            arrays = {
                "values": _create_array(
                    group, "values", values, chunk_bytes, shard_bytes
                ),
                "draw_ids": _create_array(
                    group,
                    "draw_ids",
                    ids.astype("<i8").reshape(-1, 1),
                    chunk_bytes,
                    shard_bytes,
                ),
            }
            record = {
                "id": block_id,
                "previous": current["block"],
                "start": current["count"],
                "count": len(ids),
                "first_draw": int(ids[0]),
                "last_draw": int(ids[-1]),
                "digest": digest,
                "arrays": arrays,
                "files": _seal(temporary),
            }
            write_json(temporary / "commit.json", record)
            os.replace(temporary, destination)
            sync_directory(blocks)
        finally:
            shutil.rmtree(temporary, ignore_errors=True)
        updated = {
            "block": block_id,
            "count": current["count"] + len(ids),
            "last_draw": int(ids[-1]),
            "digest": digest,
        }
        write_json(chain_path / "head.json", updated)
        return updated


class PositionWriter:
    """Buffer single draws and commit them in blocks of a bounded size."""

    def __init__(
        self,
        directory: str | Path,
        *,
        max_buffer_bytes: int = 64 * 1024**2,
        chunk_bytes: int = CHUNK_BYTES,
        shard_bytes: int = SHARD_BYTES,
    ):
        """Configure the pending-draw budget and the block geometry."""
        self.directory = Path(directory)
        self.max_buffer_bytes = max_buffer_bytes
        self._geometry = {"chunk_bytes": chunk_bytes, "shard_bytes": shard_bytes}
        self._pending: dict[int, tuple[list[int], list[Any]]] = {}
        self._bytes = 0

    def append(self, position: Any, chain_id: int, draw_id: int) -> None:
        """Buffer one draw; earlier draws are committed once the budget is full."""
        position = jax.tree.map(np.array, position)
        size = sum(leaf.nbytes for leaf in jax.tree.leaves(position)) + 256
        if self._bytes + size > self.max_buffer_bytes:
            self.flush()
        ids, draws = self._pending.setdefault(int(chain_id), ([], []))
        ids.append(int(draw_id))
        draws.append(position)
        self._bytes += size

    def flush(self) -> None:
        """Commit every pending draw."""
        for chain, (ids, draws) in self._pending.items():
            if ids:
                stacked = jax.tree.map(lambda *leaves: np.stack(leaves), *draws)
                append_block(self.directory, chain, ids, stacked, **self._geometry)
        self._pending.clear()
        self._bytes = 0

    def __enter__(self) -> PositionWriter:
        """Use the writer as a context manager."""
        return self

    def __exit__(self, kind, value, traceback) -> bool:
        """Commit pending draws after success and discard them after an error."""
        if kind is None:
            self.flush()
        self._pending.clear()
        return False


def begin_chains(directory: str | Path, chain_ids: Iterable[int], metadata=None):
    """Declare the chains a store is going to contain before writing them."""
    directory = Path(directory)
    ids = [int(chain) for chain in chain_ids]
    if not ids or len(set(ids)) != len(ids) or min(ids) < 0:
        raise ValueError("Select distinct nonnegative chain indices.")
    with lock(directory / ".chains.lock"):
        path = directory / CHAINS
        document = (
            read_json(path)
            if path.exists()
            else {"chains": {}, "metadata": metadata or {}}
        )
        if document["metadata"] != (metadata or {}):
            raise ValueError(f"Store was begun with different metadata: {directory}")
        for chain in ids:
            document["chains"].setdefault(str(chain), {"complete": False})
        write_json(path, document)


def complete_chains(directory: str | Path, chain_ids: Iterable[int]) -> None:
    """Mark chains complete; complete chains of a store hold identical draw IDs."""
    directory = Path(directory)
    with lock(directory / ".chains.lock"):
        document = read_json(directory / CHAINS)
        expected = document.get("draws")
        for chain in map(int, chain_ids):
            if str(chain) not in document["chains"]:
                raise ValueError(f"Chain {chain} was never begun in {directory}.")
            current = head(directory, chain)
            if not current["count"]:
                raise ValueError(f"Chain {chain} has no committed draws.")
            observed = {"count": current["count"], "digest": current["digest"]}
            if expected is not None and expected != observed:
                raise ValueError("Complete chains must hold identical draw IDs.")
            expected = observed
            document["chains"][str(chain)]["complete"] = True
        document["draws"] = expected
        write_json(directory / CHAINS, document)


@dataclass(frozen=True)
class _Block:
    path: Path
    start: int
    count: int
    record: dict

    def array(self, name: str):
        """Open a committed array after checking its files and geometry."""
        for relative, spec in self.record["files"].items():
            path = self.path / relative
            if not path.is_file() or path.stat().st_size != spec["size"]:
                raise ValueError(f"Missing or truncated posterior shard: {path}")
            if (
                "sha256" in spec
                and hashlib.sha256(path.read_bytes()).hexdigest() != spec["sha256"]
            ):
                raise ValueError(f"Posterior array metadata changed: {path}")
        array = zarr.open_array(str(self.path / name), mode="r")
        spec = self.record["arrays"][name]
        if list(array.shape) != spec["shape"] or array.dtype.str != spec["dtype"]:
            raise ValueError(f"Posterior array shape or dtype changed: {self.path}")
        return array


def _blocks(directory: Path, chain: int, record: dict) -> tuple[_Block, ...]:
    result, seen = [], set()
    block_id, stop = record["block"], record["count"]
    while block_id is not None:
        if block_id in seen:
            raise ValueError("Posterior block catalog contains a cycle.")
        seen.add(block_id)
        path = _chain_path(directory, chain) / "blocks" / block_id
        commit = read_json(path / "commit.json")
        if commit["id"] != block_id or commit["start"] + commit["count"] != stop:
            raise ValueError("Posterior block catalog is not contiguous.")
        result.append(_Block(path, commit["start"], commit["count"], commit))
        block_id, stop = commit["previous"], commit["start"]
    if stop != 0 or (result and result[0].record["digest"] != record["digest"]):
        raise ValueError("Posterior head does not match its committed blocks.")
    return tuple(reversed(result))


class DrawIds(Sequence):
    """Draw IDs of a committed snapshot, read lazily one block at a time."""

    def __init__(self, blocks: tuple[_Block, ...], count: int):
        """Keep block descriptors and the committed count."""
        self._blocks, self._count = blocks, count
        self._starts = tuple(block.start for block in blocks)

    def __len__(self) -> int:
        """Return the committed count without reading any IDs."""
        return self._count

    def __getitem__(self, index):
        """Read selected draw IDs."""
        if isinstance(index, slice):
            return tuple(self[i] for i in range(*index.indices(len(self))))
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        block = self._blocks[bisect.bisect_right(self._starts, index) - 1]
        return int(block.array("draw_ids")[index - block.start, 0])

    def __iter__(self):
        """Read all draw IDs block by block."""
        for block in self._blocks:
            yield from map(int, block.array("draw_ids")[:, 0])

    def __eq__(self, other) -> bool:
        """Compare with another sequence of IDs."""
        if not isinstance(other, Sequence) or len(self) != len(other):
            return False
        return all(a == b for a, b in zip(self, other, strict=True))

    def to_numpy(self) -> np.ndarray:
        """All draw IDs as one int64 vector."""
        if not self._blocks:
            return np.empty(0, dtype=np.int64)
        return np.concatenate(
            [np.asarray(block.array("draw_ids")[:, 0]) for block in self._blocks]
        )


def _selection(indices, size: int, kind: str) -> list[int]:
    values = list(range(size)) if indices is None else [int(i) for i in indices]
    if (
        not values
        or len(set(values)) != len(values)
        or min(values) < 0
        or max(values) >= size
    ):
        raise ValueError(f"Select distinct valid {kind} indices.")
    return values


@dataclass(frozen=True)
class PositionIndex:
    """A stable snapshot of committed draws; it stores block metadata, not draws."""

    directory: Path
    chain_ids: tuple[int, ...]
    draw_ids: DrawIds
    schema: dict
    blocks: tuple[tuple[_Block, ...], ...]

    @classmethod
    def open(
        cls,
        directory: str | Path,
        *,
        chains: Iterable[int] | None = None,
        require_complete: bool = True,
    ) -> PositionIndex:
        """Capture the heads of chains that hold identical draw IDs.

        ``require_complete`` demands that every declared chain was completed.
        Without it, the chains that hold draws are opened as they are, which
        serves monitoring of a running store. ``chains`` selects chain IDs.
        """
        directory = Path(directory)
        document = (
            read_json(directory / CHAINS) if (directory / CHAINS).exists() else None
        )
        chain_root = directory / "chains"
        present = (
            sorted(
                int(path.name)
                for path in chain_root.iterdir()
                if path.name.isdigit() and (path / "head.json").exists()
            )
            if chain_root.exists()
            else []
        )
        if require_complete:
            if document is None:
                raise ValueError(f"Not a completed posterior store: {directory}")
            declared = sorted(map(int, document["chains"]))
            if declared != present or any(
                not record["complete"] for record in document["chains"].values()
            ):
                raise ValueError(f"Posterior store is incomplete: {directory}")
        ids = present if chains is None else [int(chain) for chain in chains]
        if not ids or not set(ids) <= set(present):
            raise ValueError(f"Requested chains hold no draws in {directory}.")
        heads = [head(directory, chain) for chain in ids]
        first = heads[0]
        if any(
            h["count"] != first["count"] or h["digest"] != first["digest"]
            for h in heads[1:]
        ):
            raise ValueError("Chains must hold identical draw IDs.")
        blocks = tuple(
            _blocks(directory, chain, record)
            for chain, record in zip(ids, heads, strict=True)
        )
        return cls(
            directory,
            tuple(ids),
            DrawIds(blocks[0], first["count"]),
            read_json(directory / SCHEMA),
            blocks,
        )

    def load(self, chains=None, draws=None, *, on_host: bool = True) -> Any:
        """Load draws by ordinal chain and draw index.

        The returned tree has leading chain and draw axes in the requested
        order. Host loading preserves every stored bit.
        """
        chain_selection = _selection(chains, len(self.chain_ids), "chain")
        draw_selection = _selection(draws, len(self.draw_ids), "draw")
        width = self.schema["bytes_per_draw"]
        raw = np.empty((len(chain_selection), len(draw_selection), width), np.uint8)
        for out_chain, chain in enumerate(chain_selection):
            blocks = self.blocks[chain]
            starts = [block.start for block in blocks]
            by_block: dict[int, list[tuple[int, int]]] = {}
            for out_draw, draw in enumerate(draw_selection):
                block_index = bisect.bisect_right(starts, draw) - 1
                by_block.setdefault(block_index, []).append(
                    (out_draw, draw - blocks[block_index].start)
                )
            for block_index, selection in by_block.items():
                if not width:
                    continue
                block = blocks[block_index]
                array = block.array("values")
                targets = [target for target, _ in selection]
                rows = [row for _, row in selection]
                if rows == list(range(rows[0], rows[0] + len(rows))):
                    raw[out_chain, targets] = array[rows[0] : rows[0] + len(rows)]
                else:
                    raw[out_chain, targets] = array.get_orthogonal_selection(
                        (rows, slice(None))
                    )
        leaves = [
            raw[..., spec["offset"] : spec["offset"] + spec["nbytes"]]
            .view(_dtype(spec))
            .reshape(len(chain_selection), len(draw_selection), *spec["shape"])
            for spec in self.schema["leaves"]
        ]
        tree = decode_structure(self.schema["structure"], iter(leaves))
        return tree if on_host else to_device(tree)

    def iterate(self, chain: int, *, draws=None, batch_size: int = 256, on_host=True):
        """Yield consecutive batches of one ordinal chain with a leading draw axis."""
        selection = _selection(draws, len(self.draw_ids), "draw")
        for start in range(0, len(selection), batch_size):
            batch = self.load(
                [chain], selection[start : start + batch_size], on_host=on_host
            )
            yield jax.tree.map(lambda leaf: leaf[0], batch)
