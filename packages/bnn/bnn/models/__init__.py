"""Networks and their configurations."""

from bnn.models import tabular  # noqa: F401  (registration)
from bnn.models.base import (
    MODELS,
    Activation,
    ModelConfig,
    ModelSpec,
    build_module,
    register_model,
)

__all__ = [
    "MODELS",
    "Activation",
    "ModelConfig",
    "ModelSpec",
    "build_module",
    "register_model",
]
