"""A prepared dataset: arrays and a fixed partition.

A dataset is immutable once constructed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from bnn.data.config import DataConfig, Split

SPLITS_FILE = "splits.npz"


class Dataset(ABC):
    """Base of all datasets."""

    def __init__(
        self, config: DataConfig, seed: int, artifacts: str | Path | None = None
    ):
        """Load, partition and preprocess the data.

        ``seed`` drives every random choice of loading and partitioning unless
        the configuration fixes its own ``split_seed``. ``artifacts`` is the
        directory of a saved dataset; fitted preprocessing is restored from it.
        """
        self.config = config
        self.seed = seed if config.split_seed is None else config.split_seed
        self.artifacts = None if artifacts is None else Path(artifacts)
        self._generators = 0
        self.loading_permutation: np.ndarray | None = None
        path = Path(config.path).expanduser()
        self.name = path.stem
        self.data = tuple(np.asarray(array) for array in self.load_data())
        if not self.data or len({len(array) for array in self.data}) != 1:
            raise ValueError("A dataset consists of arrays with equally many rows.")
        self.splits = self.get_splits()
        self.data = tuple(np.asarray(a) for a in self.preprocess_data(self.data))

    def generator(self) -> np.random.Generator:
        """The next independent random generator of the loading sequence."""
        self._generators += 1
        return np.random.default_rng([self.seed, self._generators])

    @abstractmethod
    def load_data(self) -> tuple[np.ndarray, ...]:
        """Return aligned, unprocessed arrays with observations on the first axis."""

    def preprocess_data(self, data: tuple[np.ndarray, ...]) -> tuple[np.ndarray, ...]:
        """Transform the data after the partition has been determined."""
        return data

    def get_splits(self) -> dict[Split, np.ndarray]:
        """Row indices of the fractional train, valid and test partitions."""
        count = len(self.data[0])
        train_end = int(count * self.config.train_split)
        valid_end = int(count * (self.config.train_split + self.config.valid_split))
        train, valid, test = np.split(np.arange(count), [train_end, valid_end])
        return {Split.TRAIN: train, Split.VALID: valid, Split.TEST: test}

    def shuffle_rows(
        self, *arrays: np.ndarray, limit: int | None = None
    ) -> tuple[np.ndarray, ...]:
        """Shuffle aligned arrays, remembering the permutation of original rows."""
        self.loading_permutation = self.generator().permutation(len(arrays[0]))[:limit]
        return tuple(np.asarray(a)[self.loading_permutation] for a in arrays)

    def get_split(self, split: Split | str) -> tuple[np.ndarray, ...]:
        """The arrays of one partition."""
        indices = np.asarray(self.splits[Split(split)])
        if len(indices) and np.all(np.diff(indices) == 1):
            indices = slice(int(indices[0]), int(indices[-1]) + 1)
        return tuple(array[indices] for array in self.data)

    def size(self, split: Split | str) -> int:
        """Number of observations of a partition."""
        return len(self.splits[Split(split)])

    def model_data(self, split: Split | str = Split.TRAIN, limit: int | None = None):
        """Inputs and targets of a partition for one full-data model call."""
        return tuple(jnp.asarray(array[:limit]) for array in self.get_split(split))

    def classification_labels(self) -> np.ndarray:
        """All class indices, for validation against the model's output width."""
        return self.data[1]

    def iterate(
        self,
        split: Split | str,
        batch_size: int | None = None,
        limit: int | None = None,
    ) -> Iterator[tuple[np.ndarray, ...]]:
        """Yield a partition in order, in batches of at most ``batch_size``."""
        data = tuple(array[:limit] for array in self.get_split(split))
        count = len(data[0])
        batch_size = batch_size or max(count, 1)
        for start in range(0, count, batch_size):
            yield tuple(array[start : start + batch_size] for array in data)

    def original_splits(self) -> dict[str, np.ndarray]:
        """The partition in terms of the rows of the unshuffled source."""
        return {
            split.value: (
                indices
                if self.loading_permutation is None
                else self.loading_permutation[indices]
            )
            for split, indices in self.splits.items()
        }

    def save(self, directory: str | Path) -> None:
        """Persist the partition and everything fitted to the training data."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.savez(directory / SPLITS_FILE, **self.original_splits())

    def verify(self, directory: str | Path) -> None:
        """Require that this dataset reproduces a saved partition."""
        with np.load(Path(directory) / SPLITS_FILE) as saved:
            for name, indices in self.original_splits().items():
                if not np.array_equal(saved[name], indices):
                    raise ValueError(
                        f"The {name} partition differs from the one saved in "
                        f"{directory}: the data source or its preparation changed."
                    )

    def __str__(self) -> str:
        """Name and partition sizes."""
        sizes = ", ".join(f"{split.value}={self.size(split)}" for split in Split)
        return f"{self.name}({sizes})"
