"""The planned cells of one experiment and profile, and the commit planning them."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from runner.artifacts import ArtifactError, atomic_json, read_json
from runner.cells import CellSpec, Design, ResourceHint


@dataclass(frozen=True)
class RunManifest:
    experiment: str
    question: str
    profile: str
    seed_streams: tuple[str, ...]
    cells: tuple[CellSpec, ...]
    code: dict[str, Any]


def build_manifest(
    experiment: str, design: Design, profile: str, code: dict[str, Any]
) -> RunManifest:
    if profile not in design.profiles:
        raise ValueError(f"Profile {profile!r} is not registered for {experiment}")
    cells = tuple(
        cell for cell in design.cells if cell.factors.get("profile") == profile
    )
    if not cells:
        raise ValueError(f"{experiment} has no cells for profile {profile!r}")
    return RunManifest(
        experiment=experiment,
        question=design.question,
        profile=profile,
        seed_streams=design.seed_streams,
        cells=cells,
        code=code,
    )


def write_manifest(path: str | Path, manifest: RunManifest) -> Path:
    return atomic_json(path, asdict(manifest))


def _cell(payload: dict[str, Any]) -> CellSpec:
    resources = ResourceHint(**payload.pop("resources"))
    payload["expected_schema"] = tuple(payload["expected_schema"])
    return CellSpec(resources=resources, **payload)


def read_manifest(path: str | Path) -> RunManifest:
    raw = read_json(path)
    try:
        raw["cells"] = tuple(_cell(dict(cell)) for cell in raw["cells"])
        raw["seed_streams"] = tuple(raw["seed_streams"])
        manifest = RunManifest(**raw)
    except (KeyError, TypeError, ValueError) as error:
        raise ArtifactError(f"Invalid manifest schema: {path}") from error
    if any(cell.factors.get("profile") != manifest.profile for cell in manifest.cells):
        raise ArtifactError(f"Manifest mixes experiment profiles: {path}")
    if len({cell.cell_id for cell in manifest.cells}) != len(manifest.cells):
        raise ArtifactError(f"Manifest repeats cells: {path}")
    return manifest
