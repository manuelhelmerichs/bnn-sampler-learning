"""The data, network and posterior of a run, resolved once per process."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from bnn.config import RunConfig
from bnn.data import Task, build_dataset
from bnn.models import build_module
from bnn.pipeline.layout import RunDirectory, resolve_path
from bnn.store.trees import load_array_tree, save_array_tree
from bnn.target import ProbabilisticModel, Target
from bnn.target.contracts import validate_model_data

#: Independent random streams of a run.
STREAMS = {"network": 0, "fit": 1, "sample": 2}


def _key_seed(config: RunConfig, stream: str) -> jax.Array:
    return jax.random.fold_in(jax.random.key(config.seed), STREAMS[stream])


def load_dataset(config: RunConfig, directory: RunDirectory | None = None):
    """The dataset of a run; saved on first use and verified on every later one."""
    root = None if directory is None else directory.root
    data_config = config.data.model_copy(
        update={"path": str(resolve_path(config.data.path, root))}
    )
    saved = directory is not None and (directory.data / "splits.npz").exists()
    dataset = build_dataset(
        data_config, config.seed, artifacts=directory.data if saved else None
    )
    if directory is not None and not saved:
        dataset.save(directory.data)
    return dataset


class Context:
    """Everything a stage needs to evaluate the model of a run.

    Constructing a context loads the data and initializes the network once.
    With a run directory, the partition and immutable model collections are
    saved on first use and verified on every later one.
    """

    def __init__(self, config: RunConfig, directory: RunDirectory | None = None):
        """Resolve the dataset, the network and the probabilistic model."""
        if bool(jax.config.jax_enable_x64) != config.x64:
            raise RuntimeError(
                f"This run computes in {'double' if config.x64 else 'single'} "
                "precision; set jax_enable_x64 accordingly before creating it."
            )
        self.config = config
        self.directory = directory
        self.task: Task = config.data.task
        self.dataset = load_dataset(config, directory)
        if not self.dataset.size("train"):
            raise ValueError("This run requires a nonempty train partition.")
        self.module = build_module(config.model)
        self._input = jnp.asarray(next(self.dataset.iterate("train", batch_size=1))[0])
        variables = self.module.init(self.key("network"), self._input, train=False)
        if "batch_stats" in variables:
            raise NotImplementedError(
                "Models with mutable batch statistics are unsupported: they would "
                "have to stay fixed while sampling. Use a stateless normalization "
                "such as LayerNorm or FRN."
            )
        state = {k: v for k, v in variables.items() if k != "params"}
        if directory is not None and state:
            if directory.model_state.exists():
                state = load_array_tree(directory.model_state, on_host=False)[0]
            else:
                save_array_tree(directory.model_state, state)
        self.model_state = state
        self.model = ProbabilisticModel(
            self.module, config.prior.build(), self.task, self.model_state
        )
        validate_model_data(
            self.module,
            {**self.model_state, "params": variables["params"]},
            self.task,
            self.dataset,
        )
        template = {"params": jax.tree.map(jnp.zeros_like, variables["params"])}
        if self.task == Task.MEAN_REGRESSION:
            template["sigma"] = jnp.zeros(())
        self.position_template = template

    def key(self, stream: str) -> jax.Array:
        """Root key of one of the independent random streams of the run."""
        return _key_seed(self.config, stream)

    def network_parameters(self, chain_ids) -> Any:
        """Freshly initialized network weights of the given chains."""

        def one(chain):
            key = jax.random.fold_in(self.key("network"), chain)
            if self.model_state:
                _, variables = self.module.apply(
                    self.model_state,
                    self._input,
                    train=False,
                    rngs={"params": key},
                    mutable=["params"],
                )
            else:
                variables = self.module.init(key, self._input, train=False)
            return variables["params"]

        return jax.vmap(one)(jnp.asarray(np.asarray(chain_ids), dtype=jnp.uint32))

    def target(self) -> Target:
        """The posterior with its full training data."""
        dataset = self.dataset
        return Target(
            self.model,
            dataset.size("train"),
            lambda: {"full": dataset.model_data("train")},
        )

    def describe(self) -> str:
        """A short description for logs."""
        size = sum(leaf.size for leaf in jax.tree.leaves(self.position_template))
        return (
            f"{type(self.module).__name__} with {size} sampled parameters on "
            f"{self.dataset}, task {self.task.value}"
        )
