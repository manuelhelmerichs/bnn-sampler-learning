"""Task contracts shared by initialization, training and likelihoods."""

import jax
import jax.numpy as jnp
import numpy as np

from bnn.data.config import Task


def validate_output_shape(task: Task, outputs, targets) -> None:
    """Check static shapes before indexing, including inside compiled functions."""
    if outputs.ndim != 2 or outputs.shape[:-1] != targets.shape:
        raise ValueError(
            f"{task.value} model output shape {outputs.shape} is incompatible "
            f"with target shape {targets.shape}."
        )
    width = outputs.shape[-1]
    required = {Task.REGRESSION: 2, Task.MEAN_REGRESSION: 1}.get(task)
    if required is not None and width != required:
        raise ValueError(
            f"{task.value} requires exactly {required} model output channel(s), "
            f"got {width}."
        )
    if width < 1:
        raise ValueError("A model must have at least one output channel.")


def validate_class_labels(labels, width: int) -> None:
    """Reject invalid concrete class indices; traced values use a density guard."""
    if not jnp.issubdtype(labels.dtype, jnp.integer):
        raise ValueError("Classification targets must have an integer dtype.")
    if isinstance(labels, jax.core.Tracer):
        return
    labels = np.asarray(labels)
    if np.any(labels < 0) or np.any(labels >= width):
        raise ValueError(f"Classification targets must lie in [0, {width}).")


def validate_model_data(module, variables, task: Task, dataset) -> None:
    """Validate actual model outputs and all dataset labels before a run starts."""
    inputs, targets = next(dataset.iterate("train", batch_size=1))
    outputs = module.apply(variables, jnp.asarray(inputs), train=False)
    validate_output_shape(task, outputs, targets)
    if not np.isfinite(np.asarray(outputs)).all():
        raise ValueError("Model initialization produced nonfinite outputs.")
    if task.is_classification:
        validate_class_labels(dataset.classification_labels(), outputs.shape[-1])
