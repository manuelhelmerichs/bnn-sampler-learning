"""The fit stage and its methods: one fitted position per chain.

A method turns fresh network weights into fitted positions, where needed as a
chain program that the stage runs in resumable blocks.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

from bnn.data import Task as DataTask
from bnn.engine import Phase, Program, run_program
from bnn.fitting.config import FitConfig, ModeFit, PriorFit
from bnn.fitting.training import fitted_log_sigma
from bnn.pipeline.context import Context
from bnn.pipeline.stages.base import Stage, require_finite
from bnn.store.trees import save_array_tree
from bnn.trees import tree_index


class Fitting:
    """How one method fits the positions of a group of chains."""

    def __init__(self, config: FitConfig, context: Context):
        """Bind the method's options and the resolved run."""
        self.config = config
        self.context = context

    def program(self) -> Program | None:
        """The chain program of the method, or ``None`` if it needs no steps."""
        return None

    def initial(self, chain_ids) -> Any:
        """Input of the program: fresh network weights of every chain."""
        return self.context.network_parameters(chain_ids)

    def finish(self, result, chain_ids) -> tuple[Any, dict]:
        """Positions and metrics, both with a leading chain axis."""
        raise NotImplementedError

    def _position(self, params) -> dict:
        context = self.context
        position = {"params": params}
        if context.task == DataTask.MEAN_REGRESSION:
            x, y = context.dataset.model_data("train")
            position["sigma"] = jax.vmap(
                lambda p: fitted_log_sigma(context.model, p, x, y)
            )(params)
        return position


class FromPrior(Fitting):
    """Independent draws from the prior; the noise scale sits at its prior mode."""

    def finish(self, result, chain_ids):
        context = self.context
        keys = jax.vmap(lambda c: jax.random.fold_in(context.key("fit"), c))(
            jnp.asarray(np.asarray(chain_ids), dtype=jnp.uint32)
        )
        params = jax.vmap(context.model.prior.init)(keys, result)
        position = {"params": params}
        if context.task == DataTask.MEAN_REGRESSION:
            position["sigma"] = jnp.zeros(len(chain_ids))
        return position, {}


class PosteriorMode(Fitting):
    """Full-batch descent on the negative log posterior per observation."""

    def program(self) -> Program:
        context, config = self.context, self.config
        optimizer = config.optimizer.build()
        target = context.target()
        count = target.n_train

        def objective(params, constants):
            position = {"params": params}
            if context.task == DataTask.MEAN_REGRESSION:
                position["sigma"] = jnp.zeros(())
            return -target.log_density(position, constants) / count

        def step(carry, x, constants):
            params, state = carry
            value, gradient = jax.value_and_grad(objective)(params, constants)
            updates, state = optimizer.update(gradient, state, params)
            return (optax.apply_updates(params, updates), state), {"objective": value}

        def finish(carry, constants):
            value, gradient = jax.value_and_grad(objective)(carry[0], constants)
            return {
                "params": carry[0],
                "objective": value,
                "gradient_norm": optax.tree.norm(gradient),
            }

        return Program(
            (
                Phase(
                    "optimize",
                    config.steps,
                    lambda params, key, constants: (params, optimizer.init(params)),
                    step,
                    finish,
                ),
            ),
            target.data(),
        )

    def finish(self, result, chain_ids):
        metrics = {k: result[k] for k in ("objective", "gradient_norm")}
        return self._position(result["params"]), metrics


METHODS = {PriorFit: FromPrior, ModeFit: PosteriorMode}


def build_fitting(context: Context) -> Fitting:
    """The implementation of the configured fitting method."""
    return METHODS[type(context.config.fit)](context.config.fit, context)


class FitStage(Stage):
    """Fit one position per chain."""

    name = "fit"

    def run_task(self, run, context, task, controls) -> bool:
        method = build_fitting(context)
        program = method.program()
        result = method.initial(task.chains)
        if program is not None:
            completed, result = run_program(
                program,
                chain_ids=task.chains,
                initial=result,
                key=context.key("fit"),
                destinations=self._destinations(run, task, {}),
                pacing=self._pacing(run),
                should_stop=controls.should_stop,
                report=controls.report,
            )
            if not completed:
                return False
        positions, metrics = method.finish(result, task.chains)
        positions = jax.device_get(positions)
        require_finite(positions, task.chains, "The fitted position")
        for row, chain in enumerate(task.chains):
            save_array_tree(
                run.directory.fitted(chain),
                {"position": tree_index(positions, row)},
                metadata={
                    "method": run.config.fit.name,
                    "metrics": {
                        name: np.asarray(values)[row].tolist()
                        for name, values in metrics.items()
                    },
                },
            )
        self.mark_done(run, task)
        return True

    def finalize(self, run, context) -> None:
        missing = [
            c for c in range(run.config.chains) if not run.directory.fitted(c).exists()
        ]
        if missing:
            raise RuntimeError(f"Fitted positions of chains {missing} are missing.")
