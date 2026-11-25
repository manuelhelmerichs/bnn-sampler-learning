"""The posterior of a network: prior, likelihood and the densities built on them."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
from pydantic import model_validator

from bnn.data.config import Task
from bnn.registry import Config
from bnn.target.contracts import validate_class_labels, validate_output_shape
from bnn.target.priors import Prior, PriorDist
from bnn.trees import ParamTree, split_position


class PriorConfig(Config):
    """Prior over the network weights."""

    name: PriorDist = PriorDist.StandardNormal
    parameters: dict[str, Any] = {}

    @model_validator(mode="after")
    def _validate(self) -> PriorConfig:
        self.build()
        return self

    def build(self) -> Prior:
        """Construct the prior."""
        return self.name.get_prior(**self.parameters)


def gaussian_scale(log_scale: jax.Array) -> jax.Array:
    """Bound Gaussian scales without overflowing their reverse-mode gradients."""
    return jnp.exp(jnp.clip(log_scale, jnp.log(1e-6), jnp.log(1e6)))


class ProbabilisticModel:
    """A deterministic network with a prior and the likelihood of its task.

    A position is ``{"params": weights}``; mean regression adds the sampled log
    observation scale ``"sigma"`` with a standard normal prior.
    """

    def __init__(
        self,
        module: nn.Module,
        prior: Prior,
        task: Task,
        model_state: dict | None = None,
    ):
        """Bind the network, its prior and immutable model collections."""
        self.module = module
        self.prior = prior
        self.task = task
        self.model_state = {} if model_state is None else model_state

    def apply(self, position: ParamTree, x: jax.Array, **kwargs) -> jax.Array:
        """Evaluate the network at a position."""
        params, _ = split_position(position)
        variables = {**self.model_state, "params": params}
        return self.module.apply(variables, x, **{"train": False, **kwargs})

    def log_prior(self, position: ParamTree) -> jax.Array:
        """Log prior density of a position."""
        params, auxiliary = split_position(position)
        density = self.prior.log_prior(params)
        if "sigma" in auxiliary:
            density += jnp.sum(stats.norm.logpdf(auxiliary["sigma"]))
        return density

    def log_likelihood(
        self, position: ParamTree, x: jax.Array, y: jax.Array, **kwargs
    ) -> jax.Array:
        """Log likelihood summed over the given observations."""
        outputs = self.apply(position, x, **kwargs)
        validate_output_shape(self.task, outputs, y)
        if self.task == Task.REGRESSION:
            return jnp.sum(
                stats.norm.logpdf(
                    y, loc=outputs[..., 0], scale=gaussian_scale(outputs[..., 1])
                )
            )
        if self.task == Task.MEAN_REGRESSION:
            scale = gaussian_scale(jnp.squeeze(position["sigma"]))
            return jnp.sum(stats.norm.logpdf(y, loc=outputs[..., 0], scale=scale))
        validate_class_labels(y, outputs.shape[-1])
        log_pmf = jax.nn.log_softmax(outputs, axis=-1)
        valid = jnp.all((y >= 0) & (y < outputs.shape[-1]))
        indices = jnp.clip(y, 0, outputs.shape[-1] - 1)[..., None]
        density = jnp.sum(jnp.take_along_axis(log_pmf, indices, axis=-1))
        return jnp.where(valid, density, -jnp.inf)

    def log_density(self, position: ParamTree, x: jax.Array, y: jax.Array) -> jax.Array:
        """Unnormalized log posterior of the given observations."""
        return self.log_prior(position) + self.log_likelihood(position, x, y)

    def __str__(self) -> str:
        """Task and prior."""
        return (
            f"ProbabilisticModel(task={self.task.value}, prior={self.prior.name.value})"
        )


@dataclass(frozen=True)
class Target:
    """The posterior as seen by a chain program.

    The training data lives in the program constants, so compiled steps receive
    it as an argument instead of embedding it. ``data`` assembles those
    constants and ``log_density(position, constants)`` evaluates all
    observations.
    """

    model: ProbabilisticModel
    n_train: int
    data: Callable[[], Any]

    def log_density(self, position: ParamTree, constants: Any) -> jax.Array:
        """Full-data unnormalized log posterior."""
        x, y = constants["full"]
        return self.model.log_density(position, x, y)
