"""Which parts of a BNN force computation need to remain explicit for a shared learner to reproduce finite sampler trajectories on new architectures?"""

from pathlib import Path

from runner import Experiment

from . import (
    learned_sampler_inputs,
    width_transfer,
)
from .learned_sampler import (
    QUESTION as QUESTION,
)
from .learned_sampler import (
    bind_inputs,
    execute,
    reduce_protocol,
    suite_spec,
)


def bind_learned(spec, profile, store=None):
    return bind_inputs(
        spec, profile, None if store is None else Path(store) / "learned_sampler"
    )


def bind_width(spec, profile, store=None):
    return width_transfer.bind(
        spec, profile, None if store is None else Path(store) / "width_transfer"
    )


def prepare(profile, store):
    """Teacher tasks and fitted learners, and the per-width HMC populations."""
    store = Path(store)
    learned_sampler_inputs.prepare(profile, store / "learned_sampler")
    width_transfer.prepare(profile, store / "width_transfer")


EXPERIMENT = Experiment.join(
    QUESTION,
    {
        "learned_sampler": Experiment(
            design=suite_spec,
            measure=execute,
            summarize=reduce_protocol,
            inputs=bind_learned,
        ),
        "width_transfer": Experiment(
            design=width_transfer.design,
            measure=width_transfer.measure,
            inputs=bind_width,
        ),
    },
    prepare=prepare,
)
