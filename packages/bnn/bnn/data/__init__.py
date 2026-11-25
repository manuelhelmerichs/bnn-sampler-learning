"""Datasets: configuration, loading and partitioning."""

from __future__ import annotations

from pathlib import Path

from bnn.data.base import Dataset
from bnn.data.config import DataConfig, Split, Task


def build_dataset(
    config: DataConfig, seed: int, *, artifacts: str | Path | None = None
) -> Dataset:
    """Construct the dataset of a configuration.

    ``artifacts`` is the directory of a previously saved dataset; the result is
    then required to reproduce the saved partition.
    """
    from bnn.data.tabular import TabularData

    dataset = TabularData(config, seed, artifacts)
    if artifacts is not None:
        dataset.verify(artifacts)
    return dataset


__all__ = ["DataConfig", "Dataset", "Split", "Task", "build_dataset"]
