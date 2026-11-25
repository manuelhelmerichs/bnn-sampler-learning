"""Dataset configuration, independent of any data library."""

from __future__ import annotations

from enum import Enum

from pydantic import ConfigDict, Field, model_validator

from bnn.registry import Config


class Task(str, Enum):
    """Supervised task, which fixes the likelihood.

    ``regression`` predicts a mean and a log scale per observation,
    ``mean_regression`` predicts a mean and samples one shared log scale, and
    ``classification`` predicts class logits per observation.
    """

    REGRESSION = "regression"
    MEAN_REGRESSION = "mean_regression"
    CLASSIFICATION = "classification"

    @property
    def is_classification(self) -> bool:
        """Whether targets are class indices."""
        return self is Task.CLASSIFICATION


class Split(str, Enum):
    """Partitions of a dataset."""

    TRAIN = "train"
    VALID = "valid"
    TEST = "test"


class DataConfig(Config):
    """A local table, its task and its partition into train, valid and test."""

    model_config = ConfigDict(coerce_numbers_to_str=True)

    path: str = Field(description="Local .npy, .csv or .data file.")
    task: Task
    target_column: str | None = Field(
        default=None, description="Target column name or index; the last by default."
    )
    features: tuple[str, ...] | None = Field(
        default=None, description="Feature columns; all other columns by default."
    )
    normalize: bool = Field(
        default=False, description="Standardize with statistics of the training split."
    )
    train_split: float = Field(default=0.8, ge=0, le=1)
    valid_split: float = Field(default=0.1, ge=0, le=1)
    test_split: float = Field(default=0.1, ge=0, le=1)
    split_seed: int | None = Field(
        default=None,
        ge=0,
        description="Seed of the partition; the run seed by default.",
    )
    datapoint_limit: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _validate(self) -> DataConfig:
        fractions = (self.train_split, self.valid_split, self.test_split)
        if abs(sum(fractions) - 1.0) > 1e-6:
            raise ValueError(
                f"train, valid and test splits must sum to 1; got {sum(fractions):.8g}."
            )
        return self
