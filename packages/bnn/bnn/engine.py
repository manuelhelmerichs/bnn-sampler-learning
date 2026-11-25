"""Run chain programs in vectorized blocks with exact pause and resume.

A program is a sequence of phases. A phase is a pure transition repeated for a
fixed number of steps, for example an adaptation window, an optimizer run or
the production chain of a sampler. The engine vectorizes a phase over the
chains of one group, compiles it as a scan over blocks of steps, and after each
block commits the stored draws, the diagnostics and the complete state.

Step ``t`` of a phase only depends on the state before it, on ``t`` and on a
random key derived from ``t``. The partition into blocks therefore never
changes a result, and an interrupted program continues bit for bit from its
last commit.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from bnn.store import positions
from bnn.store.checkpoint import load_state, save_state
from bnn.store.io import read_json, write_json
from bnn.trees import tree_all_finite, tree_index


def _identity(carry: Any, constants: Any) -> Any:
    return carry


@dataclass(frozen=True)
class Phase:
    """A pure single-chain transition and what the engine keeps of it.

    Attributes:
        name: Identifies the phase in progress reports and diagnostics.
        steps: Number of transitions.
        start: ``(previous, key, constants) -> carry``. ``previous`` is the result
            of the preceding phase, or the initial value of the program.
        step: ``(carry, x, constants) -> (carry, diagnostics)``. ``x`` holds the
            step ``index``, its random ``key`` and the phase inputs of that step.
            ``diagnostics`` is a dictionary of scalars.
        finish: ``(carry, constants) -> result`` handed to the next phase.
        position: ``carry -> tree`` selecting what is stored as a draw.
        keep: Per step, whether the position after that step is stored.
        store: Name of the store receiving the kept positions.
        chain_inputs: ``(chain_ids, steps) -> {name: array}`` with leading
            chain and step axes, such as minibatch indices.
        shared_inputs: ``(steps) -> {name: array}`` with a leading step axis,
            such as a step size schedule.
        must_move: Fail if a chain never changes its position during the phase.
        check: ``carry -> tree`` of everything that must stay finite. By
            default the stored position, or the whole carry without one.
    """

    name: str
    steps: int
    start: Callable[[Any, Any, Any], Any]
    step: Callable[[Any, dict, Any], tuple[Any, dict]]
    finish: Callable[[Any, Any], Any] = _identity
    position: Callable[[Any], Any] | None = None
    keep: np.ndarray | None = None
    store: str | None = None
    chain_inputs: Callable[[np.ndarray, np.ndarray], dict] | None = None
    shared_inputs: Callable[[np.ndarray], dict] | None = None
    must_move: bool = False
    check: Callable[[Any], Any] | None = None

    def __post_init__(self) -> None:
        if self.steps < 0:
            raise ValueError(f"Phase {self.name!r} needs a nonnegative step count.")
        if self.keep is not None and (
            self.position is None or self.store is None or len(self.keep) != self.steps
        ):
            raise ValueError(
                f"Phase {self.name!r} keeps draws and needs a position, a store "
                "and one flag per step."
            )


@dataclass(frozen=True)
class Program:
    """Phases executed in order, and the arrays all of their steps read."""

    phases: tuple[Phase, ...]
    constants: Any = None


@dataclass(frozen=True)
class Pacing:
    """How the engine sizes blocks.

    A block ends with a commit. ``interval`` is the targeted time between
    commits, which bounds the work an abrupt stop can lose. ``block_steps``
    fixes the block length instead.
    """

    interval: float = 60.0
    buffer_bytes: int = 256 * 1024**2
    block_steps: int | None = None
    calibration_steps: int = 32
    max_block_steps: int = 65536


@dataclass
class Destinations:
    """Where a program stores kept draws, diagnostics and its own state."""

    state: Path
    stores: dict[str, Path] = field(default_factory=dict)
    diagnostics: Path | None = None
    chunk_bytes: int = positions.CHUNK_BYTES
    shard_bytes: int = positions.SHARD_BYTES

    def diagnostics_store(self, phase: Phase) -> Path | None:
        """Directory of the per-step diagnostics of one phase."""
        return None if self.diagnostics is None else self.diagnostics / phase.name


def _phase_keys(key, chain_ids, phase_index):
    """Independent start and step keys of each chain for one phase."""

    def one(chain):
        phase_key = jax.random.fold_in(jax.random.fold_in(key, chain), phase_index)
        return jax.random.split(phase_key)

    keys = jax.vmap(one)(jnp.asarray(chain_ids, dtype=jnp.uint32))
    return keys[:, 0], keys[:, 1]


def _start(phase, previous, start_keys, constants):
    return jax.vmap(lambda value, key: phase.start(value, key, constants))(
        previous, start_keys
    )


def _template(program, phase_index, initial, key, chain_ids):
    """Structure, shapes and dtypes of the carry of a phase, without running it."""
    previous = initial
    for index, phase in enumerate(program.phases[: phase_index + 1]):
        start_keys, _ = _phase_keys(key, chain_ids, index)
        carry = jax.eval_shape(
            lambda value, keys, phase=phase: _start(
                phase, value, keys, program.constants
            ),
            previous,
            start_keys,
        )
        if index < phase_index:
            previous = jax.eval_shape(
                jax.vmap(lambda c, phase=phase: phase.finish(c, program.constants)),
                carry,
            )
    return carry


def _block(phase: Phase, capacity: int, single: bool):
    """Compile ``length`` steps of every chain; ``capacity`` draws are kept.

    A group of one chain is compiled without a batch axis, which is markedly
    faster on CPU than a vectorized program of batch size one.
    """
    track = phase.must_move
    check = phase.check or phase.position or (lambda carry: carry)

    def one_chain(carry, moved, key, chain_xs, shared_xs, slots, constants):
        buffer = None
        if capacity:
            buffer = jax.tree.map(
                lambda leaf: jnp.zeros((capacity + 1, *leaf.shape), leaf.dtype),
                phase.position(carry),
            )

        def body(state, inputs):
            carry, buffer, moved = state
            xs, slot = inputs
            x = {**xs, "key": jax.random.fold_in(key, xs["index"])}
            before = phase.position(carry) if track else None
            carry, diagnostics = phase.step(carry, x, constants)
            if buffer is not None or track:
                position = phase.position(carry)
            if buffer is not None:
                # Steps that are not kept write to the spare last row.
                buffer = jax.tree.map(
                    lambda rows, leaf: rows.at[slot].set(leaf), buffer, position
                )
            if track:
                changed = [
                    jnp.any(new != old)
                    for new, old in zip(
                        jax.tree.leaves(position), jax.tree.leaves(before), strict=True
                    )
                ]
                moved = moved | jnp.any(jnp.stack(changed))
            return (carry, buffer, moved), diagnostics

        (carry, buffer, moved), diagnostics = jax.lax.scan(
            body, (carry, buffer, moved), ({**chain_xs, **shared_xs}, slots)
        )
        if buffer is not None:
            buffer = jax.tree.map(lambda rows: rows[:capacity], buffer)
        return carry, moved, buffer, diagnostics, tree_all_finite(check(carry))

    if not single:
        return jax.jit(jax.vmap(one_chain, in_axes=(0, 0, 0, 0, None, None, None)))

    def only_chain(carry, moved, key, chain_xs, shared_xs, slots, constants):
        first = jax.tree.map(lambda leaf: leaf[0], (carry, moved, key, chain_xs))
        result = one_chain(*first, shared_xs, slots, constants)
        return jax.tree.map(lambda leaf: leaf[None], result)

    return jax.jit(only_chain)


def simulate(phases, initial: Any, key: jax.Array, constants: Any = None) -> Any:
    """Run the phases of one chain to completion in memory.

    Nothing is stored and nothing can be resumed; the function is traceable, so
    it can be compiled and vectorized. It serves code that needs a sampler's
    dynamics without a run, such as an experiment with its own bookkeeping.
    Returns the result of the last phase.
    """
    value = initial
    for index, phase in enumerate(phases):
        start_key, step_key = jax.random.split(jax.random.fold_in(key, index))
        carry = phase.start(value, start_key, constants)
        steps = np.arange(phase.steps)
        xs = {"index": steps.astype(np.int32)}
        if phase.shared_inputs is not None:
            xs.update(phase.shared_inputs(steps))
        if phase.chain_inputs is not None:
            xs.update(tree_index(phase.chain_inputs(np.zeros(1, np.int64), steps), 0))

        def body(carry, x, phase=phase, step_key=step_key):
            x = {**x, "key": jax.random.fold_in(step_key, x["index"])}
            return phase.step(carry, x, constants)[0], None

        carry, _ = jax.lax.scan(body, carry, xs)
        value = phase.finish(carry, constants)
    return value


def _capacity(keep: np.ndarray | None, length: int) -> int:
    """Largest number of kept draws in any window of ``length`` steps."""
    if keep is None or not keep.any():
        return 0
    cumulative = np.concatenate(([0], np.cumsum(keep)))
    if length >= len(keep):
        return int(cumulative[-1])
    return int(np.max(cumulative[length:] - cumulative[:-length]))


def _power_of_two_floor(value: float) -> int:
    return 1 << max(0, int(np.floor(np.log2(max(value, 1.0)))))


class _Journal:
    """The single commit point of a chain group: state plus all store heads."""

    def __init__(self, destinations: Destinations, chain_ids):
        self.destinations = destinations
        self.chain_ids = [int(chain) for chain in chain_ids]
        self.pointer = destinations.state / "state.json"
        self.record = read_json(self.pointer) if self.pointer.exists() else None
        self.heads: dict[str, dict[str, dict]] = (
            self.record["heads"] if self.record else {}
        )

    def restore(self, program: Program) -> None:
        """Return every store this program writes to its last committed state."""
        directories = set(self.destinations.stores.values())
        for phase in program.phases:
            diagnostics = self.destinations.diagnostics_store(phase)
            if diagnostics is not None:
                directories.add(diagnostics)
        for directory in directories:
            if not Path(directory).exists():
                continue
            committed = self.heads.get(str(directory), {})
            for chain in self.chain_ids:
                positions.restore_head(
                    directory, chain, committed.get(str(chain), positions.EMPTY_HEAD)
                )

    def append(self, directory: Path, chain: int, draw_ids, tree) -> None:
        record = positions.append_block(
            directory,
            chain,
            draw_ids,
            tree,
            chunk_bytes=self.destinations.chunk_bytes,
            shard_bytes=self.destinations.shard_bytes,
        )
        self.heads.setdefault(str(directory), {})[str(chain)] = record

    def commit(self, phase_index: int, step: int, state: Any) -> None:
        name = f"state-{uuid.uuid4().hex}.npz"
        save_state(self.destinations.state / name, state)
        previous = self.record["checkpoint"] if self.record else None
        self.record = {
            "phase": phase_index,
            "step": step,
            "checkpoint": name,
            "chains": self.chain_ids,
            "heads": self.heads,
        }
        write_json(self.pointer, self.record)
        if previous is not None:
            (self.destinations.state / previous).unlink(missing_ok=True)


def run_program(
    program: Program,
    *,
    chain_ids,
    initial: Any,
    key: jax.Array,
    destinations: Destinations,
    pacing: Pacing | None = None,
    should_stop: Callable[[], bool] = lambda: False,
    report: Callable[[dict], None] = lambda event: None,
    on_phase_end: Callable[[int, Phase, Any], None] = lambda index, phase, result: None,
) -> tuple[bool, Any]:
    """Run or continue a program for one group of chains.

    ``initial`` has the chains on the leading axis of every leaf. Returns
    whether the program completed and, if so, the result of its last phase.
    ``on_phase_end`` receives the host result of every finished phase before
    the next phase starts; it may be called again for the same phase after an
    interruption, so it must be idempotent.
    """
    pacing = pacing or Pacing()
    chain_ids = np.asarray(chain_ids, dtype=np.int64)
    constants = program.constants
    destinations.state.mkdir(parents=True, exist_ok=True)
    journal = _Journal(destinations, chain_ids)
    journal.restore(program)

    if journal.record is None:
        phase_index, step = 0, 0
        start_keys, _ = _phase_keys(key, chain_ids, 0)
        carry = _start(program.phases[0], initial, start_keys, constants)
        moved = jnp.zeros(len(chain_ids), dtype=bool)
    else:
        phase_index, step = journal.record["phase"], journal.record["step"]
        template = {
            "carry": _template(program, phase_index, initial, key, chain_ids),
            "moved": jax.ShapeDtypeStruct((len(chain_ids),), np.dtype(bool)),
        }
        state, _ = load_state(
            destinations.state / journal.record["checkpoint"], template
        )
        carry, moved = jax.device_put(state["carry"]), jnp.asarray(state["moved"])

    result = None
    while phase_index < len(program.phases):
        phase = program.phases[phase_index]
        _, step_keys = _phase_keys(key, chain_ids, phase_index)
        diagnostics_store = destinations.diagnostics_store(phase)
        draw_store = destinations.stores[phase.store] if phase.store else None
        if draw_store is not None:
            positions.begin_chains(draw_store, chain_ids)
        diagnostics_begun = False

        compiled: dict[tuple[int, int], Any] = {}
        cap = pacing.max_block_steps
        if phase.keep is not None and phase.keep.any():
            draw_bytes = sum(
                int(np.prod(leaf.shape[1:])) * np.dtype(leaf.dtype).itemsize
                for leaf in jax.tree.leaves(phase.position(carry))
            )
            fraction = float(phase.keep.mean())
            cap = max(
                1,
                min(
                    cap,
                    int(
                        pacing.buffer_bytes
                        / max(1, draw_bytes * len(chain_ids))
                        / fraction
                    ),
                ),
            )
        length = pacing.block_steps or min(pacing.calibration_steps, cap)
        while step < phase.steps:
            current = min(length, phase.steps - step)
            capacity = _capacity(phase.keep, current)
            warm = (current, capacity) in compiled
            if not warm:
                compiled[(current, capacity)] = _block(
                    phase, capacity, len(chain_ids) == 1
                )
            steps = np.arange(step, step + current)
            shared = {"index": steps.astype(np.int32)}
            if phase.shared_inputs is not None:
                shared.update(phase.shared_inputs(steps))
            chain_xs = (
                phase.chain_inputs(chain_ids, steps) if phase.chain_inputs else {}
            )
            kept = (
                phase.keep[step : step + current]
                if capacity
                else np.zeros(current, dtype=bool)
            )
            slots = np.where(kept, np.cumsum(kept) - 1, capacity).astype(np.int32)

            started = time.perf_counter()
            new_carry, new_moved, buffer, diagnostics, finite = compiled[
                (current, capacity)
            ](carry, moved, step_keys, chain_xs, shared, slots, constants)
            finite = np.asarray(finite)
            elapsed = time.perf_counter() - started
            if not finite.all():
                raise FloatingPointError(
                    f"Phase {phase.name!r} produced a nonfinite state for chains "
                    f"{chain_ids[~finite].tolist()} within steps {step} to "
                    f"{step + current - 1}. Nothing of this block was stored."
                )
            carry, moved = new_carry, new_moved

            if capacity and kept.any():
                draws = jax.device_get(buffer)
                count = int(kept.sum())
                for row, chain in enumerate(chain_ids):
                    journal.append(
                        draw_store,
                        chain,
                        steps[kept],
                        jax.tree.map(
                            lambda leaf, row=row, count=count: leaf[row, :count], draws
                        ),
                    )
            if diagnostics and diagnostics_store is not None:
                if not diagnostics_begun:
                    positions.begin_chains(diagnostics_store, chain_ids)
                    diagnostics_begun = True
                values = jax.device_get(diagnostics)
                for row, chain in enumerate(chain_ids):
                    journal.append(
                        diagnostics_store, chain, steps, tree_index(values, row)
                    )
            step += current
            journal.commit(phase_index, step, {"carry": carry, "moved": moved})
            report(
                {
                    "phase": phase.name,
                    "phase_index": phase_index,
                    "phases": len(program.phases),
                    "step": step,
                    "steps": phase.steps,
                    "block": current,
                    "seconds": elapsed,
                }
            )
            if warm and pacing.block_steps is None and current == length:
                target = min(
                    cap, _power_of_two_floor(pacing.interval / (elapsed / current))
                )
                if target >= 2 * length or 2 * target <= length:
                    length = target
            if step < phase.steps and should_stop():
                return False, None

        if phase.must_move and phase.steps and not np.asarray(moved).all():
            raise FloatingPointError(
                f"Positions of chains {chain_ids[~np.asarray(moved)].tolist()} never "
                f"changed during phase {phase.name!r}. Check the starting positions, "
                "the step size and the numerical precision."
            )
        for directory in (draw_store, diagnostics_store):
            if directory is not None and Path(directory).exists():
                written = [
                    chain
                    for chain in chain_ids
                    if positions.head(directory, chain)["count"]
                ]
                if written:
                    positions.complete_chains(directory, written)
        result = jax.vmap(lambda c, phase=phase: phase.finish(c, constants))(carry)
        on_phase_end(phase_index, phase, jax.device_get(result))
        phase_index += 1
        if phase_index < len(program.phases):
            start_keys, _ = _phase_keys(key, chain_ids, phase_index)
            carry = _start(program.phases[phase_index], result, start_keys, constants)
            moved = jnp.zeros(len(chain_ids), dtype=bool)
            step = 0
            journal.commit(phase_index, 0, {"carry": carry, "moved": moved})
            if should_stop():
                return False, None
    return True, jax.device_get(result)
