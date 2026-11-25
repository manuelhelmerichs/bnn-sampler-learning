"""Is the coordinate map of a sampler worth learning?"""

from runner import Experiment

from . import campaign, network, synthetic
from .summary import summarize

EXPERIMENT = Experiment.join(
    synthetic.QUESTION,
    {
        # Bananas and funnels, whose exact Gaussianizing map is the oracle.
        "synthetic": Experiment(synthetic.design, synthetic.measure, summarize),
        # A small tanh network with a sampled reference.
        "network": Experiment(network.design, network.measure, summarize),
        # The posterior of a UCI network, without a reference: speed of burn-in.
        "campaign": Experiment(campaign.design, campaign.measure),
    },
)
