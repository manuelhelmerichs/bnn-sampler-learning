"""Fully connected networks for tabular data."""

from typing import Literal

import flax.linen as nn
import jax.numpy as jnp
from pydantic import Field, PositiveInt

from bnn.models.base import Activation, ModelConfig, register_model
from bnn.models.components import FullyConnected


class FCNConfig(ModelConfig):
    """Fully connected network."""

    name: Literal["FCN"] = "FCN"
    hidden_structure: tuple[PositiveInt, ...] = Field(
        default=(10, 10),
        min_length=1,
        description="Widths of all layers, ending with the output width.",
    )
    activation: Activation = Activation.RELU
    use_bias: bool = True


@register_model(FCNConfig)
class FCN(nn.Module):
    """Fully connected neural network."""

    config: FCNConfig

    def setup(self):
        """Create the dense stack."""
        self.fcn = FullyConnected(
            hidden_sizes=self.config.hidden_structure,
            activation=self.config.activation.flax_activation,
            use_bias=self.config.use_bias,
            last_layer_activation=None,
            blockid=None,
        )

    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        """Forward pass."""
        return self.fcn(x)
