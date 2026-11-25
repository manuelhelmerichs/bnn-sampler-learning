"""Where everything of a run lives on disk."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunDirectory:
    """The files and folders of one run.

    ::

        run.json            manifest: provenance, chain groups, stage status
        config.yaml         the resolved configuration
        data/               the partition and fitted preprocessing
        model/              immutable model collections
        fit/                one fitted position per chain, where sampling starts
        samples/            posterior draws
        warmup/             stored warmup positions, if requested
        sampling/           adapted kernel parameters and diagnostics
        work/               checkpoints and progress of unfinished chain groups
        logs/               one log per worker
    """

    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))

    @property
    def manifest(self) -> Path:
        return self.root / "run.json"

    @property
    def config(self) -> Path:
        return self.root / "config.yaml"

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def model_state(self) -> Path:
        return self.root / "model" / "state.npz"

    @property
    def fit(self) -> Path:
        return self.root / "fit"

    def fitted(self, chain: int) -> Path:
        """Fitted position of one chain."""
        return self.fit / f"chain_{int(chain)}.npz"

    @property
    def samples(self) -> Path:
        return self.root / "samples"

    @property
    def warmup(self) -> Path:
        return self.root / "warmup"

    @property
    def sampling(self) -> Path:
        return self.root / "sampling"

    def parameters(self, chain: int) -> Path:
        """Adapted kernel parameters of one chain."""
        return self.sampling / "parameters" / f"chain_{int(chain)}.npz"

    def diagnostics(self, stage: str) -> Path:
        """Root of the per-transition diagnostics of a stage, one store per phase."""
        return self.root / ("sampling" if stage == "sample" else stage) / "diagnostics"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    def work(self, stage: str, task: str) -> Path:
        """Checkpoints, progress and completion marker of one task."""
        return self.root / "work" / stage / task

    def lock(self) -> Path:
        return self.root / ".run.lock"


def resolve_path(path: str, directory: str | Path | None = None) -> Path:
    """Locate a configured input file.

    Relative paths refer to the working directory and, failing that, to the
    closest ancestor of the run directory that contains them.
    """
    candidate = Path(path).expanduser()
    if candidate.is_absolute() or candidate.exists() or directory is None:
        return candidate
    for parent in Path(directory).resolve().parents:
        if (parent / candidate).exists():
            return parent / candidate
    return candidate
