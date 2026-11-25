import importlib
import sys
import textwrap

import pytest

PACKAGE = """
from runner.cells import CellSpec, Design
from runner.experiment import Experiment


def design():
    return Design(
        question="Does the lifecycle work?",
        profiles=("smoke", "pilot", "production"),
        seed_streams=("sampling",),
        cells=tuple(
            CellSpec(
                "measure",
                {"profile": profile, "value": value},
                {"sampling": value},
                expected_schema=("metrics",),
            )
            for profile in ("smoke", "pilot", "production")
            for value in (1, 2)
        ),
    )


def measure(cell):
    return {"metrics": {"value": cell.factors["value"]}}


def summarize(manifest, results):
    return {
        "cell_metrics": [row["metrics"] for row in results],
        "total": sum(row["metrics"]["value"] for row in results),
    }


def bind(design, profile, store):
    if profile == "smoke":
        return design
    return type(design)(
        question=design.question,
        profiles=design.profiles,
        seed_streams=design.seed_streams,
        cells=tuple(
            type(cell)(
                cell.stage,
                cell.factors,
                cell.seeds,
                inputs={"prepared": {"root": str(store)}},
                expected_schema=cell.expected_schema,
            )
            if cell.factors["profile"] == profile
            else cell
            for cell in design.cells
        ),
    )


EXPERIMENT = Experiment(design, measure, summarize, inputs=bind)
"""


@pytest.fixture
def toy_package(tmp_path, monkeypatch):
    """An importable experiments package with one experiment, ``26-11-17-study``.

    The experiment imports its own module relatively, since its folder name is
    not an identifier; a private folder is not an experiment.
    """
    name = f"toy_experiments_{tmp_path.name.replace('-', '_')}"
    root = tmp_path / "source" / name
    (root / "26-11-17-study").mkdir(parents=True)
    (root / "_private").mkdir()
    (root / "__init__.py").write_text("")
    (root / "_private" / "__init__.py").write_text("")
    (root / "26-11-17-study" / "__init__.py").write_text(
        "from .study import EXPERIMENT as EXPERIMENT\n"
    )
    (root / "26-11-17-study" / "study.py").write_text(textwrap.dedent(PACKAGE))
    monkeypatch.syspath_prepend(str(tmp_path / "source"))
    yield name, importlib.import_module(f"{name}.26-11-17-study.study")
    for module in [m for m in sys.modules if m.startswith(name)]:
        del sys.modules[module]
