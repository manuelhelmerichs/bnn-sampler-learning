from typing import Any

import jax
import jax.numpy as jnp
from bnn.data.config import Task
from bnn.target.model import ProbabilisticModel
from bnn.target.priors import Prior, PriorDist

DUMMY_PRIOR = Prior(
    init=lambda rng, params: params,
    log_prior=lambda params: jnp.array(0.0),
    name=PriorDist.StandardNormal,
)


class RecordingModel:
    def __init__(self, task: Task):
        self.task = task
        self.calls: list[tuple[dict[str, Any], dict[str, Any]]] = []

    def apply(self, variables, x, **kwargs):
        self.calls.append((variables, kwargs))
        batch_size = x.shape[0]

        if self.task == Task.REGRESSION:
            return jnp.stack(
                [jnp.arange(batch_size, dtype=float), jnp.zeros(batch_size)],
                axis=-1,
            )
        if self.task == Task.MEAN_REGRESSION:
            return jnp.arange(batch_size, dtype=float)[:, None]
        return jnp.tile(jnp.array([[1.0, -1.0]]), (batch_size, 1))


def test_clipped_gaussian_scale_has_finite_forward_and_backward_values():
    import numpy as np
    from bnn.target.model import gaussian_scale

    for value in (-1000.0, -100.0, 0.0, 100.0, 1000.0):
        scale, derivative = jax.value_and_grad(gaussian_scale)(jnp.array(value))
        assert np.isfinite(scale) and np.isfinite(derivative)
        assert 0.999e-6 <= scale <= 1.001e6


def test_likelihood_gradients_remain_finite_for_extreme_log_scales():
    import numpy as np

    class Model:
        def apply(self, variables, x, **kwargs):
            return jnp.stack(
                [jnp.zeros(len(x)), jnp.full(len(x), variables["params"]["scale"])],
                axis=-1,
            )

    model = ProbabilisticModel(Model(), DUMMY_PRIOR, Task.REGRESSION)
    for value in (-100.0, 100.0, 1000.0):
        density, derivative = jax.value_and_grad(model.log_likelihood)(
            {"params": {"scale": jnp.array(value)}}, jnp.zeros((2, 1)), jnp.zeros(2)
        )
        assert np.isfinite(density)
        assert np.isfinite(derivative["params"]["scale"])


def test_likelihood_passes_frozen_state_without_sampling_it():
    recording = RecordingModel(Task.CLASSIFICATION)
    frozen = {"constants": {"embedding": jnp.ones((3, 2))}}
    model = ProbabilisticModel(
        recording, DUMMY_PRIOR, Task.CLASSIFICATION, model_state=frozen
    )
    position = {"params": {"w": jnp.zeros(1)}}
    model.log_likelihood(position, jnp.ones((1, 1)), jnp.array([0]))
    assert recording.calls[0][0]["constants"] is frozen["constants"]
    assert "constants" not in position
