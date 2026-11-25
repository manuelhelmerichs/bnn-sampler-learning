"""Tabular data with preprocessing fitted to the training partition only."""

from pathlib import Path

import numpy as np

from bnn.data.base import Dataset
from bnn.data.config import DataConfig, Split, Task

PREPROCESSING_FILE = "preprocessing.npz"


class TabularData(Dataset):
    """Numeric tables with selected columns and optional standardization."""

    def __init__(
        self, config: DataConfig, seed: int, artifacts: str | Path | None = None
    ):
        """Load the table; saved normalization statistics are restored."""
        self.preprocessing: dict[str, np.ndarray] = {}
        saved = None if artifacts is None else Path(artifacts) / PREPROCESSING_FILE
        self.preprocessing_path = (
            saved if saved is not None and saved.exists() else None
        )
        super().__init__(config, seed, artifacts)

    def load_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Load a finite numeric table; column selectors are indices or CSV header names."""
        path = Path(self.config.path).expanduser()
        suffix = path.suffix.lstrip(".")
        selectors = [self.config.target_column, *(self.config.features or [])]
        named = any(
            s is not None and not str(s).lstrip("-").isdigit() for s in selectors
        )
        names = None
        if named:
            if suffix != "csv":
                raise ValueError("Named column selection requires a CSV header.")
            table = np.genfromtxt(
                path, delimiter=",", names=True, dtype=float, encoding="utf-8"
            )
            names = list(table.dtype.names or ())
            data = np.column_stack([np.atleast_1d(table[name]) for name in names])
        elif suffix == "npy":
            data = np.load(path, allow_pickle=False, mmap_mode="r")
        elif suffix in {"csv", "data"}:
            data = np.loadtxt(path, delimiter="," if suffix == "csv" else None, ndmin=2)
        else:
            raise ValueError("Tabular data must be a .npy, .csv, or .data file.")
        if data.ndim != 2 or data.shape[1] < 2 or not len(data):
            raise ValueError(
                "Tabular data must contain observations and at least two columns."
            )
        if (
            not np.issubdtype(data.dtype, np.number)
            or np.iscomplexobj(data)
            or not np.isfinite(data).all()
        ):
            raise ValueError("Tabular data must contain only finite numeric values.")

        def column(selector):
            if names is not None and str(selector) in names:
                return names.index(str(selector))
            try:
                index = int(selector)
            except (ValueError, TypeError) as error:
                raise ValueError(f"Unknown column {selector!r}.") from error
            if not -data.shape[1] <= index < data.shape[1]:
                raise ValueError(f"Column {selector!r} is out of range.")
            return index % data.shape[1]

        target = column(
            self.config.target_column if self.config.target_column is not None else -1
        )
        features = (
            [column(s) for s in self.config.features]
            if self.config.features is not None
            else [i for i in range(data.shape[1]) if i != target]
        )
        if not features or len(set(features)) != len(features) or target in features:
            raise ValueError(
                "Features must be nonempty, unique, and exclude the target column."
            )
        self.selected_columns = np.asarray([*features, target], dtype=np.int32)
        (data,) = self.shuffle_rows(data, limit=self.config.datapoint_limit)
        x, y = data[:, features], data[:, target]
        if self.config.task == Task.CLASSIFICATION:
            if not np.all(
                (y >= 0) & (y == np.floor(y)) & (y <= np.iinfo(np.int32).max)
            ):
                raise ValueError(
                    "Classification targets must be nonnegative integer class indices."
                )
            y = y.astype(np.int32)
        x = x.astype(np.float32)
        if not np.isfinite(x).all():
            raise ValueError("Features cannot be represented as finite float32 values.")
        return x, y

    def preprocess_data(self, data):
        """Fit or restore training statistics, then transform every split."""
        x, y = data
        if not self.config.normalize:
            return data
        if self.preprocessing_path is not None:
            with np.load(self.preprocessing_path, allow_pickle=False) as saved:
                self.preprocessing = {key: saved[key] for key in saved.files}
            if not np.array_equal(self.preprocessing["columns"], self.selected_columns):
                raise ValueError(
                    "Saved preprocessing columns do not match this dataset."
                )
            if str(self.preprocessing["task"]) != self.config.task.value:
                raise ValueError(
                    "Saved preprocessing task does not match this dataset."
                )
        else:
            train = self.splits[Split.TRAIN]
            if len(train) == 0:
                raise ValueError(
                    "Fitting normalization requires nonempty training data."
                )
            x_scale = x[train].std(axis=0)
            self.preprocessing = {
                "columns": self.selected_columns,
                "task": np.asarray(self.config.task.value),
                "x_mean": np.asarray(x[train].mean(axis=0)),
                "x_scale": np.where(x_scale > 0, x_scale, 1.0),
            }
            if self.config.task != Task.CLASSIFICATION:
                y_scale = y[train].std()
                self.preprocessing.update(
                    y_mean=np.asarray(y[train].mean()),
                    y_scale=np.asarray(np.where(y_scale > 0, y_scale, 1.0)),
                )
        for key in ("x_mean", "x_scale"):
            if (
                self.preprocessing[key].shape != (x.shape[1],)
                or not np.isfinite(self.preprocessing[key]).all()
            ):
                raise ValueError(f"Invalid saved preprocessing {key}.")
        if np.any(self.preprocessing["x_scale"] <= 0):
            raise ValueError("Normalization scales must be positive.")
        x = (x - self.preprocessing["x_mean"]) / self.preprocessing["x_scale"]
        if self.config.task != Task.CLASSIFICATION:
            if (
                not np.isfinite(self.preprocessing["y_mean"]).all()
                or not np.isfinite(self.preprocessing["y_scale"]).all()
                or np.any(self.preprocessing["y_scale"] <= 0)
            ):
                raise ValueError("Invalid target normalization statistics.")
            y = (y - self.preprocessing["y_mean"]) / self.preprocessing["y_scale"]
        return x, y

    def save(self, directory: str | Path) -> None:
        """Persist the partition and the fitted normalization statistics."""
        super().save(directory)
        if self.preprocessing:
            np.savez(Path(directory) / PREPROCESSING_FILE, **self.preprocessing)
