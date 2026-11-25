"""The sample stage: run the sampler of every chain and store its draws."""

from __future__ import annotations

import jax

from bnn.engine import Program, run_program
from bnn.pipeline.run import Run
from bnn.pipeline.stages.base import Stage, require_finite
from bnn.samplers import SAMPLES, WARMUP
from bnn.store.positions import PositionIndex, begin_chains
from bnn.store.trees import save_array_tree
from bnn.trees import tree_index


class SampleStage(Stage):
    """Run the sampler of every chain and store its draws."""

    name = "sample"

    def _stores(self, run: Run) -> dict:
        stores = {SAMPLES: run.directory.samples}
        if run.config.sampler.keep_warmup:
            stores[WARMUP] = run.directory.warmup
        return stores

    def prepare(self, run, context) -> None:
        for directory in self._stores(run).values():
            begin_chains(directory, range(run.config.chains))

    def run_task(self, run, context, task, controls) -> bool:
        sampler = run.config.sampler
        target = context.target()
        phases = sampler.phases(target)
        initial = jax.device_put(run.fitted(task.chains))

        def on_phase_end(index, phase, result):
            # The input of the sampling phase carries the adapted parameters.
            if index != len(phases) - 2:
                return
            parameters = sampler.parameters(result)
            require_finite(parameters, task.chains, "The adapted sampler parameters")
            for row, chain in enumerate(task.chains):
                save_array_tree(
                    run.directory.parameters(chain), tree_index(parameters, row)
                )

        completed, _ = run_program(
            Program(phases, target.data()),
            chain_ids=task.chains,
            initial=initial,
            key=context.key("sample"),
            destinations=self._destinations(run, task, self._stores(run)),
            pacing=self._pacing(run),
            should_stop=controls.should_stop,
            report=controls.report,
            on_phase_end=on_phase_end,
        )
        if completed:
            self.mark_done(run, task)
        return completed

    def finalize(self, run, context) -> None:
        index = PositionIndex.open(run.directory.samples)
        if list(index.chain_ids) != list(range(run.config.chains)):
            raise RuntimeError("The posterior store does not hold every chain.")
