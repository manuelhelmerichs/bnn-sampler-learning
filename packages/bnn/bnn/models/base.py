"""The model family: configuration base and registry."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar

from flax import linen as nn

from bnn.registry import Family, Registry


class Activation(str, Enum):
    """Activation function of hidden layers."""

    SIGMOID = "sigmoid"
    RELU = "relu"
    GELU = "gelu"
    TANH = "tanh"
    SOFTMAX = "softmax"
    LEAKY_RELU = "leaky_relu"
    IDENTITY = "identity"

    @property
    def flax_activation(self) -> Callable:
        """The corresponding Flax function."""
        if self is Activation.IDENTITY:
            return lambda x: x
        return getattr(nn, self.value)


class ModelConfig(Family):
    """Base configuration of a network; ``name`` selects the architecture."""

    members: ClassVar[Registry] = Registry("model")


@dataclass(frozen=True)
class ModelSpec:
    """A registered architecture."""

    config: type[ModelConfig]
    module: type[nn.Module]


MODELS: Registry[ModelSpec] = Registry("model")


def register_model(config: type[ModelConfig]):
    """Class decorator registering a Flax module under its configuration's name."""

    def register(module: type[nn.Module]) -> type[nn.Module]:
        name = config.model_fields["name"].default
        ModelConfig.members.register(name, config)
        MODELS.register(name, ModelSpec(config, module))
        return module

    return register


def build_module(config: ModelConfig) -> nn.Module:
    """Instantiate the network of a configuration."""
    return MODELS[config.name].module(config=config)
