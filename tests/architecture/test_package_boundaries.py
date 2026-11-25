"""Installed packages must declare the dependencies their code imports."""

import ast
import importlib.util
import re
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
LIBRARIES = ("bnn", "runner")
IMPORT_DISTRIBUTIONS = {"yaml": "pyyaml"}


def _imports(path):
    relative = path.relative_to(ROOT).with_suffix("").parts
    if relative[0] == "packages":
        relative = relative[2:]
    module = ".".join(relative)
    package = (
        module.removesuffix(".__init__")
        if path.stem == "__init__"
        else module.rpartition(".")[0]
    )
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            yield from ((node.lineno, name.name) for name in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                module = importlib.util.resolve_name("." * node.level + module, package)
            yield node.lineno, module
            yield from ((node.lineno, f"{module}.{alias.name}") for alias in node.names)


def _declared(project):
    return {
        re.sub(r"[-_.]+", "-", re.match(r"[\w.-]+", dep)[0]).lower()
        for dep in [
            *project.get("dependencies", []),
            *(
                d
                for deps in project.get("optional-dependencies", {}).values()
                for d in deps
            ),
        ]
    }


@pytest.mark.parametrize("package", [*sorted(LIBRARIES), "experiments"])
def test_direct_imports_have_declared_dependencies(package):
    if package == "experiments":
        config, folder = ROOT / "pyproject.toml", ROOT / "experiments"
    else:
        config = ROOT / "packages" / package / "pyproject.toml"
        folder = ROOT / "packages" / package / package
    declared = _declared(tomllib.loads(config.read_text())["project"])
    violations = []
    for path in folder.rglob("*.py"):
        if "tests" in path.relative_to(folder).parts:
            continue
        for line, module in _imports(path):
            name = module.partition(".")[0]
            if name in sys.stdlib_module_names or name == package:
                continue
            distribution = IMPORT_DISTRIBUTIONS.get(name, name.replace("_", "-"))
            if distribution.lower() not in declared:
                violations.append(f"{path.relative_to(ROOT)}:{line}: {module}")
    assert not violations, (
        "Declare direct dependencies (including extras):\n" + "\n".join(violations)
    )


def test_workspace_layers_and_experiments_stay_independent():
    """Declaring a forbidden dependency must not make it an allowed dependency.

    An experiment imports only its own modules, the libraries and third parties.
    """
    allowed = {
        "bnn": {"bnn"},
        "runner": {"runner", "bnn"},
        "experiments": {"experiments", "runner", "bnn"},
    }
    roots = [
        *[(name, ROOT / "packages" / name / name) for name in LIBRARIES],
        ("experiments", ROOT / "experiments"),
    ]
    violations = []
    for owner, folder in roots:
        for path in folder.rglob("*.py"):
            parts = path.relative_to(folder).parts
            # Library fixtures can bridge APIs; scientific implementations cannot.
            if owner in LIBRARIES and "tests" in parts:
                continue
            for line, module in _imports(path):
                target = module.split(".")[0]
                forbidden = target in allowed and target not in allowed[owner]
                if owner == "experiments" and target == "experiments":
                    target_parts = module.split(".")
                    if len(target_parts) > 1:
                        forbidden |= len(parts) < 2 or target_parts[1] != parts[0]
                if forbidden:
                    violations.append(f"{path.relative_to(ROOT)}:{line}: {module}")
    assert not violations, "Forbidden boundary imports:\n" + "\n".join(violations)


def test_bnn_library_stays_below_its_pipeline():
    """Only the pipeline, the run configuration and the command line import each other.

    Everything else in ``bnn`` is library code that experiments use without a run.
    """
    package = ROOT / "packages" / "bnn" / "bnn"
    upper = ("bnn.pipeline", "bnn.config", "bnn.cli")
    violations = []
    for path in package.rglob("*.py"):
        if path.relative_to(package).parts[0] in {
            "pipeline",
            "config.py",
            "cli.py",
            "__main__.py",
        }:
            continue
        violations.extend(
            f"{path.relative_to(ROOT)}:{line}: {module}"
            for line, module in _imports(path)
            if any(module == name or module.startswith(f"{name}.") for name in upper)
        )
    assert not violations, "Library code imports the pipeline:\n" + "\n".join(
        violations
    )
