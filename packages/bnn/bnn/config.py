"""The one configuration of a run, and loading it from YAML.

A run is described by a single document: the data, the model and its prior,
how its chains are fitted, and optionally how they are sampled. Component sections select a registered member
by ``name`` and accept exactly that member's options.
"""

from __future__ import annotations

import copy
import importlib
import itertools
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, NonNegativeInt, PositiveFloat, PositiveInt, model_validator

from bnn.data.config import DataConfig
from bnn.fitting.config import FitConfig, PriorFit
from bnn.models import ModelConfig
from bnn.registry import Config
from bnn.samplers import SamplerConfig
from bnn.target import PriorConfig

STAGES = ("fit", "sample")
#: The section of the configuration that requests each stage.
SECTIONS = {"fit": "fit", "sample": "sampler"}


class ExecutionConfig(Config):
    """How a run is executed. None of these options changes the target."""

    group_size: PositiveInt | None = Field(
        default=None,
        description="Chains compiled into one vectorized program. By default "
        "one chain on CPU and all chains otherwise.",
    )
    checkpoint_interval: PositiveFloat = Field(
        default=60.0, description="Targeted seconds between commits of a chain group."
    )
    block_steps: PositiveInt | None = Field(
        default=None, description="Fixed transitions per commit instead of a time."
    )
    buffer_bytes: PositiveInt = Field(
        default=256 * 1024**2, description="Draws held in memory between commits."
    )
    chunk_bytes: PositiveInt = 1024**2
    shard_bytes: PositiveInt = 16 * 1024**2
    diagnostics: bool = Field(
        default=True, description="Store per-transition scalar diagnostics."
    )

    @model_validator(mode="after")
    def _validate(self) -> ExecutionConfig:
        if self.chunk_bytes > self.shard_bytes:
            raise ValueError("chunk_bytes must not exceed shard_bytes.")
        return self


class RunConfig(Config):
    """Everything that defines one run."""

    name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.=+-]*$")
    group: str = Field(
        default="",
        pattern=r"^([A-Za-z0-9][A-Za-z0-9_.=+-]*(/[A-Za-z0-9][A-Za-z0-9_.=+-]*)*)?$",
        description="Folder of related runs below the run root.",
    )
    seed: NonNegativeInt = 0
    x64: bool = Field(default=False, description="Compute in double precision.")
    chains: PositiveInt = Field(
        default=4,
        description="Independent chains: the members of the fit, and with a "
        "sampler the chains that start from them.",
    )
    imports: tuple[str, ...] = Field(
        default=(),
        description="Modules to import first; they register additional "
        "models, samplers or fitting methods.",
    )
    data: DataConfig
    model: ModelConfig
    prior: PriorConfig = PriorConfig()
    fit: FitConfig = PriorFit()
    sampler: SamplerConfig | None = Field(
        default=None,
        description="Without a sampler the run ends with its fitted positions.",
    )
    execution: ExecutionConfig = ExecutionConfig()

    @model_validator(mode="before")
    @classmethod
    def _import_extensions(cls, value: Any) -> Any:
        # Component names can only be resolved once their modules registered them.
        if isinstance(value, dict):
            for module in value.get("imports") or ():
                importlib.import_module(str(module))
        return value

    def requests(self, stage: str) -> bool:
        """Whether the configuration asks for a stage."""
        return getattr(self, SECTIONS[stage]) is not None

    @property
    def relative_directory(self) -> Path:
        """Location of the run below the run root."""
        return Path(self.group) / self.name

    def identity(self, stage: str) -> dict[str, Any]:
        """The options a stage's results depend on."""
        document = self.to_dict()
        # The data file is identified by its content, not by its location.
        del document["data"]["path"]
        keys = ["seed", "x64", "chains", "data", "model", "prior", "fit"]
        if stage == "sample":
            keys.append("sampler")
        return {key: document[key] for key in keys}


def _set(document: dict, dotted: str, value: Any) -> None:
    *parents, leaf = dotted.split(".")
    node = document
    for key in parents:
        node = node.setdefault(key, {})
        if not isinstance(node, dict):
            raise ValueError(f"Cannot set {dotted!r}: {key!r} is not a section.")
    node[leaf] = value


def _label(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.+-]+", "-", str(value)).strip("-") or "x"


def expand(document: dict) -> list[dict]:
    """Expand the optional ``sweep`` section into one document per combination.

    ``sweep`` maps dotted option paths to lists of values. Every combination
    becomes a run named after the base name and its swept values.
    """
    document = copy.deepcopy(document)
    sweep = document.pop("sweep", None)
    if not sweep:
        return [document]
    if not isinstance(sweep, dict) or not all(
        isinstance(values, list) and values for values in sweep.values()
    ):
        raise ValueError("sweep maps option paths to nonempty lists of values.")
    documents = []
    for combination in itertools.product(*sweep.values()):
        variant = copy.deepcopy(document)
        for dotted, value in zip(sweep, combination, strict=True):
            _set(variant, dotted, value)
        suffix = "_".join(
            f"{dotted.rsplit('.', 1)[-1]}={_label(value)}"
            for dotted, value in zip(sweep, combination, strict=True)
        )
        variant["name"] = f"{document.get('name', 'run')}_{suffix}"
        documents.append(variant)
    return documents


def load_configs(
    path: str | Path, overrides: dict[str, Any] | None = None
) -> list[RunConfig]:
    """Read a YAML file, or every YAML file of a directory, into run configurations.

    ``overrides`` maps dotted option paths to values that replace the file's.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No configuration at {path}.")
    files = (
        sorted(p for p in path.iterdir() if p.suffix in {".yaml", ".yml"})
        if path.is_dir()
        else [path]
    )
    configs = []
    for file in files:
        document = yaml.safe_load(file.read_text())
        if not isinstance(document, dict):
            raise ValueError(f"{file} does not contain a configuration mapping.")
        for dotted, value in (overrides or {}).items():
            _set(document, dotted, value)
        try:
            configs.extend(RunConfig.model_validate(item) for item in expand(document))
        except ValueError as error:
            raise ValueError(f"{file}: {error}") from error
    if not configs:
        raise ValueError(f"No configurations found in {path}.")
    locations = [config.relative_directory for config in configs]
    if len(set(locations)) != len(locations):
        raise ValueError("Runs must have distinct group and name combinations.")
    return configs


def dump_config(config: RunConfig) -> str:
    """Serialize a configuration as YAML."""
    return yaml.safe_dump(config.to_dict(), sort_keys=False)
