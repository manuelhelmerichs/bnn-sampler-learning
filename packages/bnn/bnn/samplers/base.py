"""The sampler family: shared options and the contract of a member.

A sampler is a configuration class that turns a posterior into the phases of a
chain program. Registering a subclass is all it takes to add one; the engine
supplies vectorization over chains, storage, checkpoints and resumption.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import numpy as np
from pydantic import Field, NonNegativeInt, PositiveInt

from bnn.registry import Family, Registry

if TYPE_CHECKING:
    from bnn.engine import Phase
    from bnn.target import Target

SAMPLES = "samples"
WARMUP = "warmup"


class SamplerConfig(Family):
    """Options shared by all samplers; ``name`` selects the algorithm."""

    members: ClassVar[Registry] = Registry("sampler")

    warmup: NonNegativeInt = Field(
        default=1000, description="Transitions of adaptation before sampling."
    )
    steps: PositiveInt = Field(default=1000, description="Sampling transitions.")
    thinning: PositiveInt = Field(
        default=1, description="Store every this many sampling transitions."
    )
    keep_warmup: bool = Field(
        default=False, description="Also store the positions visited during warmup."
    )

    def phases(self, target: Target) -> tuple[Phase, ...]:
        """Phases of one chain, starting from an initial position."""
        raise NotImplementedError

    def stored_draws(self) -> np.ndarray:
        """Transition indices of the stored sampling draws."""
        return np.arange(0, self.steps, self.thinning)

    def keep(self) -> np.ndarray:
        """Per sampling transition, whether its position is stored."""
        return np.arange(self.steps) % self.thinning == 0
