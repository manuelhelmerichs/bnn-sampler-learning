"""The single experiment contract: a design, a measurement and a summary."""

from __future__ import annotations

import functools
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from runner.cells import CellSpec, Design
from runner.manifest import RunManifest
from runner.paths import relative

Binder = Callable[[Design, str, Path | None], Design]


def reduce_protocol(
    manifest: RunManifest, results: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    """The default summary: cell metrics, stage counts and resources."""
    if len(results) != len(manifest.cells):
        raise ValueError("Reducer received an incomplete manifest")
    stages = Counter(cell.stage for cell in manifest.cells)
    resource_rows = [result.get("resources", {}) for result in results]
    resources = {}
    for key in ("peak_rss_mb", "accelerator_peak_memory_mb", "payload_bytes"):
        values = [row[key] for row in resource_rows if row.get(key) is not None]
        resources[key] = (
            {"total": float(sum(values)), "maximum": float(max(values))}
            if values
            else None
        )
    return {
        "question": manifest.question,
        "stage_cell_counts": dict(sorted(stages.items())),
        "estimands": sorted(
            {estimand for result in results for estimand in result.get("estimands", [])}
        ),
        "cell_metrics": [result["metrics"] for result in results],
        "resource_measurements": resources,
        "publishable": manifest.profile == "production",
    }


def _joined_design(question: str, parts: tuple[tuple[str, Experiment], ...]) -> Design:
    return _combine(question, [part.design() for _, part in parts])


def _combine(question: str, designs: Sequence[Design]) -> Design:
    return Design(
        question=question,
        profiles=designs[0].profiles,
        seed_streams=tuple(
            dict.fromkeys(
                stream for design in designs for stream in design.seed_streams
            )
        ),
        cells=tuple(cell for design in designs for cell in design.cells),
    )


@dataclass(frozen=True)
class Experiment:
    """One scientific question.

    ``design`` returns the cells of every profile, ``measure`` runs one cell and
    ``summarize`` reduces every cell of a manifest. Optional hooks: ``prepare``
    builds experiment-owned inputs in a store, and ``inputs`` binds that store
    into the planned cells.
    """

    design: Callable[[], Design]
    measure: Callable[[CellSpec], dict[str, Any]] | None = None
    summarize: Callable[[RunManifest, Sequence[dict[str, Any]]], dict[str, Any]] = (
        reduce_protocol
    )
    inputs: Binder | None = None
    prepare: Callable[[str, Path], Any] | None = None
    parts: Mapping[str, Experiment] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (self.measure is None) == (not self.parts):
            raise ValueError("An experiment has either a measurement or parts")

    @classmethod
    def join(
        cls, question: str, parts: Mapping[str, Experiment], **hooks: Any
    ) -> Experiment:
        """Combine parts with disjoint stages; each binds, measures and summarizes its cells."""
        if len(parts) < 2:
            raise ValueError("A joined experiment needs at least two parts")
        joined = cls(
            design=functools.partial(_joined_design, question, tuple(parts.items())),
            parts=dict(parts),
            **hooks,
        )
        joined._owners()
        return joined

    def _owners(self) -> dict[str, tuple[str, Experiment]]:
        owners = self.__dict__.get("_stage_owners")
        if owners is None:
            owners = {}
            for name, part in self.parts.items():
                for stage in {cell.stage for cell in part.design().cells}:
                    if stage in owners:
                        raise ValueError(
                            f"Parts {owners[stage][0]} and {name} share stage {stage}"
                        )
                    owners[stage] = (name, part)
            object.__setattr__(self, "_stage_owners", owners)
        return owners

    def _part(self, cell: CellSpec) -> Experiment:
        try:
            return self._owners()[cell.stage][1]
        except KeyError:
            raise ValueError(f"Unregistered measurement stage: {cell.stage}") from None

    def bind(self, profile: str, store: Path | None = None) -> Design:
        """The design with inputs bound for ``profile``."""
        if (
            profile != "smoke"
            and store is not None
            and self.prepare is not None
            and not Path(store).is_dir()
        ):
            raise FileNotFoundError(f"Missing prepared inputs: {store}")
        return self._bind(self.design(), profile, store)

    def _bind(self, design: Design, profile: str, store: Path | None) -> Design:
        if not self.parts:
            return (
                design if self.inputs is None else self.inputs(design, profile, store)
            )
        bound = []
        for name, part in self.parts.items():
            stages = {
                stage for stage, (owner, _) in self._owners().items() if owner == name
            }
            selected = replace(
                design, cells=tuple(c for c in design.cells if c.stage in stages)
            )
            bound.append(part._bind(selected, profile, store))
        return _combine(design.question, bound)

    def execute(self, cell: CellSpec) -> dict[str, Any]:
        if self.parts:
            return self._part(cell).execute(cell)
        return self.measure(cell)

    def reduce(
        self, manifest: RunManifest, results: Sequence[dict[str, Any]]
    ) -> dict[str, Any]:
        if len(results) != len(manifest.cells):
            raise ValueError("Every manifest cell needs exactly one result")
        if not self.parts:
            return self.summarize(manifest, results)
        for cell in manifest.cells:
            self._part(cell)
        summary = reduce_protocol(manifest, results)
        parts = {}
        for name, part in self.parts.items():
            pairs = [
                (cell, result)
                for cell, result in zip(manifest.cells, results, strict=True)
                if self._owners()[cell.stage][0] == name
            ]
            if not pairs:
                continue
            cells, rows = zip(*pairs, strict=True)
            reduced = part.reduce(replace(manifest, cells=cells), rows)
            reduced.pop("cell_metrics", None)  # Kept once, in manifest order.
            summary.setdefault("row_summaries", {}).update(
                reduced.pop("row_summaries", {})
            )
            parts[name] = {"consumed_cell_ids": [c.cell_id for c in cells], **reduced}
        summary["parts"] = parts
        summary["accepted"] = all(p.get("accepted", True) for p in parts.values())
        summary["publishable"] = summary["publishable"] and all(
            p.get("publishable", True) for p in parts.values()
        )
        return summary


def bind_prepared(
    design: Design, profile: str, store: Path | None, folder: Callable[[CellSpec], str]
) -> Design:
    """Point the ``prepared`` input of each ``profile`` cell at ``store / folder(cell)``.

    Only cells that declare a ``prepared`` input are bound. Smoke cells use
    synthetic inputs and stay unbound. A missing store fails; binding never
    creates inputs.
    """
    if profile == "smoke":
        return design
    store = Path(store).resolve()
    if not store.is_dir():
        raise FileNotFoundError(f"Missing prepared inputs: {store}")
    return replace(
        design,
        cells=tuple(
            replace(
                cell,
                cell_id="",
                inputs={
                    **cell.inputs,
                    "prepared": {
                        **cell.inputs["prepared"],
                        "root": relative(store / folder(cell)),
                    },
                },
            )
            if cell.factors.get("profile") == profile and "prepared" in cell.inputs
            else cell
            for cell in design.cells
        ),
    )
