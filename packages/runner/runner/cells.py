"""Immutable experiment designs and cell declarations."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from runner.paths import repository_root
from runner.serialization import fingerprint

if TYPE_CHECKING:
    from bnn import Run

# Where ``bnn run`` puts the runs of ``configs/``, relative to the repository.
RUNS = "runs"


@dataclass(frozen=True)
class ResourceHint:
    cpu: int = 1
    gpu: int = 0
    memory_gb: float = 2.0
    walltime_minutes: int = 10

    def __post_init__(self) -> None:
        if self.cpu < 1 or self.gpu < 0 or self.memory_gb <= 0:
            raise ValueError("Resource values must be positive (GPU may be zero)")
        if self.walltime_minutes < 1:
            raise ValueError("walltime_minutes must be positive")


@dataclass(frozen=True)
class CellSpec:
    """One measurement: its stage, factors, seeds and inputs.

    An input is either a ``bnn`` configuration file, given relative to the
    repository root such as ``configs/trajectory/uci_energy_mile_8x100000.yaml``,
    whose finished run the cell reads, or a mapping whose ``root`` is a folder
    the experiment prepared. The cell id names the cell's folder; it follows
    from everything else.
    """

    stage: str
    factors: dict[str, Any]
    seeds: dict[str, int]
    inputs: dict[str, Any] = field(default_factory=dict)
    expected_schema: tuple[str, ...] = ()
    resources: ResourceHint = field(default_factory=ResourceHint)
    role: str = "confirmation"
    cell_id: str = ""

    def __post_init__(self) -> None:
        if not self.stage:
            raise ValueError("A cell stage is required")
        if self.role not in {"selection", "confirmation", "descriptive", "pilot"}:
            raise ValueError(f"Unknown cell role: {self.role}")
        for name, value in self.inputs.items():
            if isinstance(value, str) and Path(value).suffix not in {".yaml", ".yml"}:
                raise ValueError(f"Input {name!r} must name a bnn configuration file")
        identity = {
            "stage": self.stage,
            "factors": self.factors,
            "seeds": self.seeds,
            "inputs": self.inputs,
            "role": self.role,
        }
        generated = f"{self.stage}-{fingerprint(identity, namespace='cell')[:16]}"
        if self.cell_id and self.cell_id != generated:
            raise ValueError("Cell identity disagrees with its configuration")
        object.__setattr__(self, "cell_id", generated)

    def path(self, name: str) -> Path:
        """The folder of input ``name``: a checked run, or a prepared folder."""
        value = self.inputs[name]
        if isinstance(value, str):
            return self.run(name).directory.root
        return repository_root() / value["root"]

    def run(self, name: str) -> Run:
        """The run of the configuration ``name``, checked against that file."""
        from bnn import run_of

        root = repository_root()
        return run_of(root / self.inputs[name], root / RUNS)


@dataclass(frozen=True)
class Design:
    """The question, profiles, seed streams and cells of one experiment."""

    question: str
    profiles: tuple[str, ...]
    seed_streams: tuple[str, ...]
    cells: tuple[CellSpec, ...]

    def __post_init__(self) -> None:
        if not self.question:
            raise ValueError("A design states its question")
        if len(set(self.seed_streams)) != len(self.seed_streams):
            raise ValueError("Seed stream names must be unique")
        ids = [cell.cell_id for cell in self.cells]
        if len(ids) != len(set(ids)):
            raise ValueError("A design cannot contain duplicate cells")
