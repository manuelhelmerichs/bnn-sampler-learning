"""How a run fits one position per chain.

Without a sampler the fitted positions are the result of the run; with one,
every chain starts from its fitted position.
"""

from __future__ import annotations

import math
from typing import Any, ClassVar, Literal

from pydantic import Field, PositiveInt, model_validator

from bnn.registry import Config, Family, Registry


class OptimizerConfig(Config):
    """An Optax optimizer with an optional learning-rate schedule.

    The objective contains the prior, so the optimizer has no regularization
    of its own.
    """

    name: Literal["adam", "sgd"] = "adam"
    parameters: dict[str, Any] = Field(
        default={"learning_rate": 1e-3}, description="Optimizer arguments."
    )
    schedule: Literal["cosine_decay", "warmup_cosine_decay", "linear"] | None = None
    schedule_parameters: dict[str, Any] = Field(
        default={}, description="Arguments of the Optax schedule."
    )

    @model_validator(mode="after")
    def _validate(self) -> OptimizerConfig:
        if self.schedule is None and self.schedule_parameters:
            raise ValueError("schedule_parameters requires a schedule.")
        value = self.parameters.get("learning_rate")
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError("learning_rate must be finite and nonnegative.")
        self.build()
        return self

    def build(self):
        """Construct the optimizer."""
        import optax

        optimizers = {"adam": optax.adam, "sgd": optax.sgd}
        schedules = {
            "cosine_decay": optax.cosine_decay_schedule,
            "warmup_cosine_decay": optax.warmup_cosine_decay_schedule,
            "linear": optax.linear_schedule,
        }
        parameters = dict(self.parameters)
        if self.schedule is not None:
            parameters["learning_rate"] = schedules[self.schedule](
                **self.schedule_parameters
            )
        return optimizers[self.name](**parameters)


class FitConfig(Family):
    """Base of the fitting methods; ``name`` selects one."""

    members: ClassVar[Registry] = Registry("fit")


@FitConfig.member("prior")
class PriorFit(FitConfig):
    """Draw every chain's position from the prior."""

    name: Literal["prior"] = "prior"


@FitConfig.member("mode")
class ModeFit(FitConfig):
    """Full-batch minimization of the negative log posterior.

    Mean regression holds the log noise scale at its prior mode zero while
    optimizing, and then pairs every mode with the scale that maximizes its
    training likelihood.
    """

    name: Literal["mode"] = "mode"
    optimizer: OptimizerConfig = OptimizerConfig()
    steps: PositiveInt = 1000
