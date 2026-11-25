"""How much can a metric gain a sampler, and what may one metric evaluation cost?"""

from runner import Experiment

from . import benchmark, ladder
from .summary import summarize

EXPERIMENT = Experiment.join(
    benchmark.QUESTION,
    {
        # Synthetic families with an exact Gaussianizing map.
        "headroom": Experiment(benchmark.design, benchmark.measure, summarize),
        # A deep-linear posterior along its width ladder.
        "ladder": Experiment(ladder.design, ladder.measure, summarize),
    },
)
