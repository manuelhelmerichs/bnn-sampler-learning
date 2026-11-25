"""Priors for Bayesian Neural Networks."""

import math
from enum import Enum
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
from jax.flatten_util import ravel_pytree

from bnn.trees import ParamTree, PRNGKey, split_key_by_tree


class PriorDist(str, Enum):
    """Supported prior families. StandardNormal accepts no parameters."""

    NORMAL = "Normal"
    StandardNormal = "StandardNormal"

    def get_prior(self, **parameters):
        """Return a prior class."""
        return Prior.from_name(self, **parameters)


class Prior(NamedTuple):
    """Base Class for Priors."""

    init: Callable[[PRNGKey, ParamTree], ParamTree]
    log_prior: Callable[[ParamTree], jax.Array]
    name: PriorDist

    @classmethod
    def from_name(cls, name: PriorDist, **parameters):
        """Initialize the prior class instance."""
        if "scale" in parameters and (
            not math.isfinite(parameters["scale"]) or parameters["scale"] <= 0
        ):
            raise ValueError("Prior scale must be finite and positive")
        if "loc" in parameters and not math.isfinite(parameters["loc"]):
            raise ValueError("Prior loc must be finite")

        def build(*, name, init, log_prior):
            def validate_precision(params):
                for leaf in jax.tree.leaves(params):
                    bounds = jnp.finfo(leaf.dtype)
                    lower = 16 * math.sqrt(float(bounds.tiny))
                    upper = math.sqrt(float(bounds.max)) / 16
                    value = parameters.get("scale", 1.0)
                    if not lower <= value <= upper:
                        raise ValueError(
                            f"Prior scale is not representable in {leaf.dtype}; expected [{lower}, {upper}]"
                        )

            def initialize(key, params):
                validate_precision(params)
                return init(key, params)

            def density(params):
                validate_precision(params)
                return log_prior(params)

            return cls(name=name, init=initialize, log_prior=density)

        if name == PriorDist.StandardNormal:
            if parameters:
                raise ValueError(
                    "StandardNormal takes no parameters; use Normal for a location or scale"
                )
            return build(
                name=PriorDist.StandardNormal,
                init=init_normal(),
                log_prior=log_prior_normal(),
            )
        elif name == PriorDist.NORMAL:
            return build(
                name=PriorDist.NORMAL,
                init=init_normal(**parameters),
                log_prior=log_prior_normal(**parameters),
            )
        raise NotImplementedError(f"Prior Distribution {name} not implemented.")


def init_normal(
    loc: float = 0.0, scale: float = 1.0
) -> Callable[[PRNGKey, ParamTree], jax.Array]:
    """Initialize from Normal distribution."""

    @jax.jit
    def rng_normal(leaf: jax.Array, rng: PRNGKey) -> jax.Array:
        sample = jax.random.normal(rng, shape=jnp.shape(leaf), dtype=jnp.dtype(leaf))
        return sample * scale + loc

    def init(rng: PRNGKey, params: ParamTree) -> jax.Array:
        rng_tree = split_key_by_tree(rng, params)
        return jax.tree.map(rng_normal, params, rng_tree)

    return init


def log_prior_normal(
    loc: float = 0.0, scale: float = 1.0
) -> Callable[[ParamTree], jax.Array]:
    """Evaluate Normal prior on all weights."""

    def log_prior(params: ParamTree) -> jax.Array:
        scores = stats.norm.logpdf(ravel_pytree(params)[0], loc=loc, scale=scale)
        return jnp.sum(scores)

    return log_prior
